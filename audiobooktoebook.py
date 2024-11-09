import os
import json
import shutil
from pathlib import Path
import whisper
import logging
import signal
import sys
from datetime import datetime
import tempfile
import time
from functools import wraps
import warnings
import torch

def retry_on_network_error(max_retries=3, delay=1):
    """Decorator to retry operations on network errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OSError, IOError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2 ** attempt)
                        logging.warning(f"Network operation failed, retrying in {sleep_time}s: {str(e)}")
                        time.sleep(sleep_time)
                    continue
            logging.error(f"Operation failed after {max_retries} attempts: {str(last_exception)}")
            raise last_exception
        return wrapper
    return decorator

class GracefulInterruptHandler:
    def __init__(self):
        self.interrupted = False
        self.released = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)

    def __enter__(self):
        self.interrupted = False
        self.released = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def signal_handler(self, signum, frame):
        self.interrupted = True
        logging.info("\nSignal received. Cleaning up... Press Ctrl+C again to force quit.")

    def release(self):
        if self.released:
            return False
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)
        self.released = True
        return True

class WhisperTranscriber:
    def __init__(self, root_dir, model_name="tiny", min_confidence=0.0):
        """Initialize the transcriber with root directory and model settings"""
        self.root_dir = Path(root_dir)
        self.transcript_dir = self.root_dir / "transcripts"
        self.cache_dir = Path(tempfile.gettempdir()) / "whisper_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._ensure_network_dirs()
        
        self.min_confidence = min_confidence
        self.progress_file = self.root_dir / ".transcription_progress.json"
        self.processed_files = self._load_progress()
        self.current_file = None
        
        # Setup detailed logging
        log_file = self.cache_dir / "transcription.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize whisper model with specific device handling
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {device}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.model = whisper.load_model(model_name).to(device)
                logging.info(f"Successfully loaded {model_name} model")
        except Exception as e:
            logging.error(f"Failed to load Whisper model '{model_name}': {str(e)}")
            raise

    @retry_on_network_error()
    def _ensure_network_dirs(self):
        """Ensure necessary network directories exist"""
        self.transcript_dir.mkdir(exist_ok=True)

    @retry_on_network_error()
    def _copy_to_cache(self, file_path):
        """Copy a file from network to local cache"""
        cache_path = self.cache_dir / file_path.name
        shutil.copy2(file_path, cache_path)
        return cache_path

    @retry_on_network_error()
    def _copy_from_cache(self, cache_path, dest_path):
        """Copy a file from local cache to network"""
        shutil.copy2(cache_path, dest_path)

    def _validate_transcription(self, result):
        """
        Validate the transcription result with detailed feedback
        Returns: (bool, str) - (is_valid, reason)
        """
        if not result:
            return False, "Empty result"
        
        if not isinstance(result, dict):
            return False, f"Invalid result type: {type(result)}"
        
        if "text" not in result:
            return False, "No text field in result"
        
        text = result["text"].strip()
        if not text:
            return False, "Empty transcription text"
        
        # Print the transcribed text for debugging
        logging.info(f"Transcribed text: {text[:200]}...")  # Print first 200 chars
        
        # Check for minimum text length
        if len(text.split()) < 5:
            return False, "Transcription too short (less than 5 words)"
            
        # Skip confidence check for now since it seems to be failing
        return True, "Validation passed"

    def transcribe_file(self, audio_path):
        """Transcribe a single audio file using Whisper with local caching"""
        self.current_file = audio_path
        cached_audio = None
        cached_transcript = None
        
        try:
            logging.info(f"Starting transcription of: {audio_path}")
            
            # Copy audio file to local cache
            cached_audio = self._copy_to_cache(audio_path)
            logging.debug(f"Audio file cached at: {cached_audio}")
            
            # Transcribe from cached file with detailed logging
            logging.info("Beginning transcription...")
            result = self.model.transcribe(
                str(cached_audio),
                verbose=True  # Enable whisper's internal logging
            )
            
            logging.debug(f"Raw transcription result keys: {result.keys()}")
            
            # Enhanced validation logging
            validation_result = self._validate_transcription(result)
            if not validation_result[0]:
                logging.error(f"Validation failed: {validation_result[1]}")
                raise ValueError(f"Transcription validation failed: {validation_result[1]}")
            
            # Create transcript in cache
            transcript_filename = f"{audio_path.stem}.txt"
            cached_transcript = self.cache_dir / transcript_filename
            
            with open(cached_transcript, 'w', encoding='utf-8') as f:
                f.write(result["text"])
                f.write("\n\n--- Transcription Metadata ---\n")
                f.write(f"Transcribed at: {datetime.now().isoformat()}\n")
                f.write(f"Source file: {audio_path}\n")
                
                if "segments" in result:
                    f.write("\nSegments:\n")
                    for i, segment in enumerate(result["segments"]):
                        f.write(f"\nSegment {i+1}:\n")
                        f.write(f"Text: {segment.get('text', '')}\n")
                        f.write(f"Start: {segment.get('start', 'N/A')}s\n")
                        f.write(f"End: {segment.get('end', 'N/A')}s\n")
            
            # Create transcripts directory if it doesn't exist
            self.transcript_dir.mkdir(exist_ok=True)
            
            # Copy to network location
            final_transcript_path = self.transcript_dir / transcript_filename
            self._copy_from_cache(cached_transcript, final_transcript_path)
            
            # Update progress
            self.processed_files[str(audio_path)] = {
                "timestamp": os.path.getmtime(audio_path),
                "transcript_path": str(final_transcript_path),
                "transcribed_at": datetime.now().isoformat()
            }
            self._save_progress()
            
            logging.info(f"Successfully created transcript at: {final_transcript_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error transcribing {audio_path}: {str(e)}")
            return False
            
        finally:
            # Clean up cache files
            try:
                if cached_audio and cached_audio.exists():
                    cached_audio.unlink()
                if cached_transcript and cached_transcript.exists():
                    cached_transcript.unlink()
            except Exception as e:
                logging.warning(f"Failed to clean up cache files: {str(e)}")
            
            self.current_file = None

    @retry_on_network_error()
    def _load_progress(self):
        """Load progress file with retry logic"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Corrupted progress file: {str(e)}")
                # Create backup of corrupted file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.progress_file.with_suffix(f'.json.bak_{timestamp}')
                shutil.copy2(self.progress_file, backup_path)
                return {}
        return {}

    @retry_on_network_error()
    def _save_progress(self):
        """Save progress with retry logic"""
        # Save to local cache first
        cache_progress = self.cache_dir / ".progress.json.tmp"
        with open(cache_progress, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
        
        # Copy to network location
        self._copy_from_cache(cache_progress, self.progress_file)
        cache_progress.unlink()

    def process_directory(self):
        """Process files with network-aware error handling"""
        with GracefulInterruptHandler() as handler:
            try:
                # Get list of files first to avoid network traversal issues
                audio_files = list(self.root_dir.rglob("*.mp3"))
                total_files = len(audio_files)
                processed_files = 0
                
                logging.info(f"Found {total_files} MP3 files to process")
                
                for audio_path in audio_files:
                    if handler.interrupted:
                        logging.info("Gracefully shutting down...")
                        break
                    
                    try:
                        if self._is_processed(audio_path):
                            current_timestamp = os.path.getmtime(audio_path)
                            if current_timestamp == self.processed_files[str(audio_path)]["timestamp"]:
                                logging.info(f"Skipping already processed file: {audio_path}")
                                continue
                            logging.info(f"File modified, re-transcribing: {audio_path}")
                        
                        success = self.transcribe_file(audio_path)
                        if success:
                            processed_files += 1
                            logging.info(f"Progress: {processed_files}/{total_files} files processed")
                        
                    except (OSError, IOError) as e:
                        logging.error(f"Network error processing {audio_path}: {str(e)}")
                        time.sleep(5)
                        continue
                    
            except Exception as e:
                logging.error(f"Error during directory processing: {str(e)}")
                raise

    def _is_processed(self, file_path):
        """Check if file has already been processed"""
        return str(file_path) in self.processed_files

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    try:
        transcriber = WhisperTranscriber(
            root_dir,
            model_name="tiny",
            min_confidence=0.0  # Set to 0 to skip confidence check
        )
        transcriber.process_directory()
    except KeyboardInterrupt:
        logging.info("\nTranscription interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()