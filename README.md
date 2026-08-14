# SongForge

> AI-powered song cover generation from imperfect source recordings

## The Problem

You have an old recording of an original song. The vocals are buried, the mix is rough, and modern AI cover tools can't detect the melody. SongForge bridges the gap:

1. **Separates** vocals from instruments using state-of-the-art source separation (Demucs)
2. **Transcribes** the isolated vocals using Whisper to verify against known lyrics
3. **Enhances** the vocal track (volume, EQ, de-noise)
4. **Generates** a new cover using AI music generation (MMX/MiniMax)
5. **Mixes** the result with the original instrumental for a polished cover

## Quick Start

```bash
pip install -r requirements.txt

# Cover a song from an imperfect recording
python -m songforge cover \
  --input song.mp3 \
  --lyrics "your lyrics here" \
  --style "acoustic indie folk, warm intimate vocals" \
  --output cover.mp3

# Just separate stems
python -m songforge separate --input song.mp3 --output-dir stems/

# Just transcribe
python -m songforge transcribe --input vocals.wav

# Compare transcription against known lyrics (for cover verification)
python -m songforge transcribe --input vocals.wav --compare lyrics.txt

# Spectral precheck — diagnose a recording BEFORE separation
python -m songforge analyze --input song.mp3

# Enhance vocal quality (volume, EQ, optional de-noise)
python -m songforge enhance --input vocals.wav --output enhanced.wav --volume 3.0 --eq-freq 2000 --denoise
```

## Requirements

- Python 3.10+
- ffmpeg
- Demucs (auto-installed)
- MMX CLI (for cover generation)
- Whisper (for transcription)

## License

MIT
