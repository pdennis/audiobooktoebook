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
                        sleep_time = delay * (2 ** attempt)  # Exponential backoff
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
    def __init__(self, root_dir, model_name="base", min_confidence=0.5):
        """
        Initialize the transcriber with root directory and model settings
        
        Args:
            root_dir (str): Root directory to start crawling from
            model_name (str): Whisper model to use (tiny, base, small, medium, large)
            min_confidence (float): Minimum confidence threshold for transcription (0-1)
        """
        self.root_dir = Path(root_dir)
        self.transcript_dir = self.root_dir / "transcripts"
        
        # Create local cache directory
        self.cache_dir = Path(tempfile.gettempdir()) / "whisper_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Ensure network directories exist
        self._ensure_network_dirs()
        
        self.min_confidence = min_confidence
        self.progress_file = self.root_dir / ".transcription_progress.json"
        self.processed_files = self._load_progress()
        self.current_file = None
        
        # Setup logging
        log_file = self.cache_dir / "transcription.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize whisper model
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.model = whisper.load_model(model_name)
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
        """Validate the transcription result"""
        if not result or not isinstance(result, dict):
            return False
        
        if "text" not in result or not result["text"].strip():
            return False
            
        # Check average segment confidence if available
        if "segments" in result:
            confidences = [seg.get("confidence", 0) for seg in result["segments"]]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < self.min_confidence:
                    logging.warning(f"Low confidence transcription: {avg_confidence:.2f}")
                    return False
        
        return True

    def transcribe_file(self, audio_path):
        """Transcribe a single audio file using Whisper with local caching"""
        self.current_file = audio_path
        temp_transcript = None
        cached_audio = None
        
        try:
            logging.info(f"Transcribing: {audio_path}")
            
            # Copy audio file to local cache
            cached_audio = self._copy_to_cache(audio_path)
            logging.info(f"Copied to local cache: {cached_audio}")
            
            # Transcribe from cached file
            result = self.model.transcribe(str(cached_audio))
            
            # Validate transcription result
            if not self._validate_transcription(result):
                raise ValueError("Transcription validation failed")
            
            # Create transcript in cache first
            transcript_filename = f"{audio_path.stem}.txt"
            cached_transcript = self.cache_dir / transcript_filename
            
            with open(cached_transcript, 'w') as f:
                f.write(result["text"])
                # Add metadata
                f.write("\n\n--- Transcription Metadata ---\n")
                f.write(f"Transcribed at: {datetime.now().isoformat()}\n")
                f.write(f"Source file: {audio_path}\n")
                
                if "segments" in result:
                    confidences = [seg.get("confidence", 0) for seg in result["segments"]]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        f.write(f"Average confidence score: {avg_confidence:.2f}\n")
            
            # Copy transcript to network location
            final_transcript_path = self.transcript_dir / transcript_filename
            self._copy_from_cache(cached_transcript, final_transcript_path)
            
            # Update progress
            self.processed_files[str(audio_path)] = {
                "timestamp": os.path.getmtime(audio_path),
                "transcript_path": str(final_transcript_path),
                "transcribed_at": datetime.now().isoformat(),
                "avg_confidence": avg_confidence if 'avg_confidence' in locals() else None
            }
            self._save_progress()
            
            logging.info(f"Transcription complete: {final_transcript_path}")
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
    # Get the directory from command line argument or use current directory
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    try:
        transcriber = WhisperTranscriber(
            root_dir,
            model_name="base",
            min_confidence=0.5
        )
        transcriber.process_directory()
    except KeyboardInterrupt:
        logging.info("\nTranscription interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()