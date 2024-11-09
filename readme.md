# AudiobookToEbook

A Python script that recursively processes audio files in a directory structure, using OpenAI's Whisper to transcribe them into text. Perfect for converting lecture recordings, audiobooks, or other spoken content into readable text.

## Features

- Recursive directory crawling for `.mp3` files
- Smart progress tracking to avoid re-processing files
- Network drive support with local caching
- Robust error handling and interruption recovery
- Confidence scoring for transcription quality
- Detailed logging and metadata

## Prerequisites

- Python 3.7+
- FFmpeg

## Installation

1. Install FFmpeg (on macOS):
```bash
brew install ffmpeg
```

2. Install Python dependencies:
```bash
pip install openai-whisper torch logging pathlib
```

For Apple Silicon Macs (M1/M2/M3), first install PyTorch:
```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## Usage

1. Clone or download this repository
2. Edit the `root_dir` in the script to point to your audio files directory:
```python
root_dir = "/path/to/your/audiofiles"
```

3. Run the script:
```bash
python transcriber.py
```

The script will:
- Create a `transcripts` directory in your root folder
- Process all `.mp3` files recursively
- Save transcripts as `.txt` files
- Track progress in `.transcription_progress.json`

## Output Structure

```
root_dir/
├── audiobook1.mp3
├── folder1/
│   └── audiobook2.mp3
├── transcripts/
│   ├── audiobook1.txt
│   └── audiobook2.txt
└── .transcription_progress.json
```

## Features for Network Drives

- Local caching of audio files before processing
- Automatic retries for network operations
- Exponential backoff for network errors
- Safe file operations with atomic writes

## Configuration Options

Modify these parameters in the script to customize behavior:

```python
transcriber = WhisperTranscriber(
    root_dir,
    model_name="base",     # Options: tiny, base, small, medium, large
    min_confidence=0.5     # Minimum confidence threshold (0-1)
)
```

## Logging

The script creates a detailed log file in:
- Local drives: `root_dir/transcription.log`
- Network drives: `[temp_dir]/whisper_cache/transcription.log`

## Interrupting

- Press Ctrl+C once for graceful shutdown
- Press Ctrl+C twice to force quit
- Progress is saved automatically

## Metadata

Each transcript includes metadata:
- Transcription timestamp
- Source file information
- Confidence scores
- Processing details

## Troubleshooting

1. **Network Issues**: Check network connection and permissions
2. **Memory Issues**: Try using a smaller Whisper model
3. **FFmpeg Missing**: Install FFmpeg using package manager
4. **Corrupted Progress**: Check `.transcription_progress.json.bak` backups


