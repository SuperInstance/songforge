"""
SongForge full cover pipeline: Separate → Enhance → Transcribe → Generate → Mix
"""

import shutil
import subprocess
from pathlib import Path
from .separate import separate_stems
from .enhance import enhance_vocals
from .transcribe import transcribe_audio
from .analyze import analyze_recording, diagnose_vocal_presence, format_report

def cover_pipeline(args):
    """Run the full cover generation pipeline."""
    keep_stems = getattr(args, "keep_stems", False)
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

    # Intermediates are only kept when the user asks for them
    if not keep_stems:
        _cleanup_intermediates(stems, enhanced)

    print(f"\n✅ Cover complete: {final}")
    return final


def _cleanup_intermediates(stems: dict, enhanced: str) -> None:
    """Remove the stems and enhanced vocal wav this pipeline run created.

    Honors the CLI's --keep-stems flag: by default SongForge leaves no
    intermediate files behind. Only the per-song Demucs output directory
    (and the model dir if it becomes empty) plus the enhanced wav are
    removed — sibling song folders are never touched.

    Safety: paths are resolved before removal and anything that resolves
    to the current working directory or one of its ancestors is refused.
    This keeps a relative path like "vocals.wav" from ever escalating into
    deleting the project directory.
    """
    cwd = Path.cwd().resolve()

    def _refuse_if_unsafe(resolved: Path, what: str) -> None:
        if resolved == cwd or cwd.is_relative_to(resolved):
            raise ValueError(
                f"Refusing to remove {resolved} ({what}): would delete the working directory"
            )

    for path in (stems["vocals"], stems["no_vocals"]):
        song_dir = Path(path).resolve().parent
        if not song_dir.exists():
            continue
        _refuse_if_unsafe(song_dir, "stem dir")
        shutil.rmtree(song_dir)
        # Drop the model dir (e.g. htdemucs/) if this was its last song
        model_dir = song_dir.parent
        if model_dir.exists() and not any(model_dir.iterdir()):
            _refuse_if_unsafe(model_dir, "model dir")
            model_dir.rmdir()
    enhanced_path = Path(enhanced).resolve()
    if enhanced_path.is_file():
        _refuse_if_unsafe(enhanced_path, "enhanced vocal wav")
        enhanced_path.unlink()
    print("  Removed intermediate stems and enhanced vocals (use --keep-stems to retain)")

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
