"""
SongForge full cover pipeline: Separate → Enhance → Transcribe → Generate → Mix
"""

import subprocess
from pathlib import Path
from .separate import separate_stems
from .enhance import enhance_vocals
from .transcribe import transcribe_audio
from .analyze import analyze_recording, diagnose_vocal_presence, format_report

def cover_pipeline(args):
    """Run the full cover generation pipeline."""
    print("=" * 60)
    print("SongForge — Cover Pipeline")
    print("=" * 60)
    
    # Step 0: Spectral precheck
    print("\n[0/5] Running spectral precheck...")
    report = analyze_recording(args.input)
    diagnosis = diagnose_vocal_presence(report)
    print(format_report(report, diagnosis))
    
    if diagnosis["recommendation"] == "skip_separation" and not getattr(args, "force", False):
        print("\n⚠️  Separation skipped — vocals below noise floor.")
        print("  Use --force to attempt separation anyway.")
        print("  Proceeding directly to lyric-matched generation...\n")
        # Skip to generation with known lyrics
        cover_file = _generate_cover(args.style, args.lyrics, args.output)
        print(f"\n✅ Cover complete: {cover_file}")
        return cover_file
    
    # Step 1: Separate
    print("\n[1/5] Separating stems...")
    stems = separate_stems(args.input, output_dir="./stems/")
    vocals = stems["vocals"]
    instrumental = stems["no_vocals"]
    
    # Step 2: Enhance vocals
    print("\n[2/5] Enhancing vocals...")
    enhanced = enhance_vocals(vocals, output_file="enhanced_vocals.wav", volume=3.0, denoise=True)
    
    # Step 3: Transcribe
    print("\n[3/5] Transcribing to verify...")
    result = transcribe_audio(enhanced, model="small")
    print(f"  Detected: {result['transcription'][:100]}...")
    
    # Step 4: Generate cover via MMX
    print("\n[4/5] Generating cover with MMX...")
    cover_file = _generate_cover(args.style, args.lyrics, args.output)
    
    # Step 5: Mix
    print("\n[5/5] Mixing cover with instrumental...")
    final = _mix_tracks(cover_file, instrumental, args.output.replace(".mp3", "_mixed.mp3"))
    
    print(f"\n✅ Cover complete: {final}")
    return final

def _generate_cover(style: str, lyrics: str, output: str) -> str:
    """Generate cover using MMX."""
    result = subprocess.run([
        "mmx", "music", "generate",
        "--prompt", style,
        "--out-dir", str(Path(output).parent),
        "--out-prefix", Path(output).stem,
        "--yes", "--quiet", "--non-interactive"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  MMX generate failed, trying cover mode...")
        result = subprocess.run([
            "mmx", "music", "cover",
            "--audio-file", "enhanced_vocals.wav",
            "--lyrics", lyrics,
            "--prompt", style,
            "--output", output
        ], capture_output=True, text=True)
    
    return output

def _mix_tracks(vocals: str, instrumental: str, output: str) -> str:
    """Mix vocals with instrumental."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", vocals,
        "-i", instrumental,
        "-filter_complex", "amix=inputs=2:duration=longest",
        output
    ], capture_output=True, text=True)
    return output
