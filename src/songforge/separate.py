"""Vocal separation using Demucs."""

import sys
import subprocess
from pathlib import Path

def separate_stems(input_file: str, output_dir: str = "./stems/", model: str = "htdemucs") -> dict:
    """Separate audio into vocals and no_vocals stems using Demucs."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Separating stems from {input_path.name} using {model}...")
    
    result = subprocess.run([
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-o", str(output_path),
        str(input_path)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr}")
    
    stem_dir = output_path / model / input_path.stem
    vocals_file = stem_dir / "vocals.wav"
    no_vocals_file = stem_dir / "no_vocals.wav"
    
    if not vocals_file.exists():
        raise FileNotFoundError(f"Vocals not found at {vocals_file}")
    
    print(f"  Vocals: {vocals_file}")
    print(f"  No vocals: {no_vocals_file}")
    
    return {"vocals": str(vocals_file), "no_vocals": str(no_vocals_file)}
