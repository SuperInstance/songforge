"""Vocal transcription using Whisper."""

from pathlib import Path

def transcribe_audio(input_file: str, model: str = "small", compare: str = None) -> dict:
    """Transcribe audio using OpenAI Whisper."""
    import whisper
    
    print(f"Loading Whisper model '{model}'...")
    whisper_model = whisper.load_model(model)
    
    print(f"Transcribing {input_file}...")
    result = whisper_model.transcribe(input_file)
    
    transcription = result.get("text", "").strip()
    print(f"\nTranscription:\n{transcription}\n")
    
    output = {
        "transcription": transcription,
        "language": result.get("language", "unknown"),
        "segments": len(result.get("segments", [])),
    }
    
    if compare:
        compare_path = Path(compare)
        if compare_path.exists():
            known = compare_path.read_text().strip()
            output["known_lyrics"] = known
            trans_words = set(transcription.lower().split())
            known_words = set(known.lower().split())
            if known_words:
                overlap = len(trans_words & known_words) / len(known_words)
                output["lyrics_overlap"] = round(overlap, 2)
                print(f"Lyrics overlap: {output['lyrics_overlap']:.0%}")
    
    return output
