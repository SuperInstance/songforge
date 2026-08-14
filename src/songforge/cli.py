#!/usr/bin/env python3
"""
SongForge CLI — cover songs from imperfect recordings.

Usage:
  python -m songforge cover --input song.mp3 --lyrics "..." --style "..." --output cover.mp3
  python -m songforge separate --input song.mp3 --output-dir stems/
  python -m songforge transcribe --input vocals.wav
  python -m songforge enhance --input vocals.wav --output enhanced.wav
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(prog="songforge", description="AI song cover tool")
    subparsers = parser.add_subparsers(dest="command")
    
    # Cover command
    cover = subparsers.add_parser("cover", help="Full pipeline")
    cover.add_argument("--input", "-i", required=True, help="Input audio file")
    cover.add_argument("--lyrics", "-l", required=True, help="Known lyrics text")
    cover.add_argument("--style", "-s", default="acoustic folk, warm vocals", help="Musical style prompt")
    cover.add_argument("--output", "-o", default="cover.mp3", help="Output file")
    cover.add_argument("--keep-stems", action="store_true", help="Keep intermediate files")
    cover.add_argument("--force", action="store_true", help="Force separation even if precheck warns")
    
    # Separate command
    sep = subparsers.add_parser("separate", help="Separate vocals from instruments")
    sep.add_argument("--input", "-i", required=True)
    sep.add_argument("--output-dir", "-o", default="./stems/")
    sep.add_argument("--model", "-m", default="htdemucs")
    
    # Transcribe command
    trans = subparsers.add_parser("transcribe", help="Transcribe vocals using Whisper")
    trans.add_argument("--input", "-i", required=True)
    trans.add_argument("--model", "-m", default="small")
    trans.add_argument("--compare", "-c", help="Compare against known lyrics file")
    
    # Analyze command
    ana = subparsers.add_parser("analyze", help="Spectral precheck — diagnose recording before separation")
    ana.add_argument("--input", "-i", required=True, help="Input audio file")
    ana.add_argument("--duration", "-d", type=float, default=30.0, help="Seconds to analyze (default 30)")
    
    # Enhance command
    enh = subparsers.add_parser("enhance", help="Enhance vocal quality")
    enh.add_argument("--input", "-i", required=True)
    enh.add_argument("--output", "-o", default="enhanced.wav")
    enh.add_argument("--volume", type=float, default=3.0, help="Volume boost in dB")
    enh.add_argument("--eq-freq", type=int, default=2000, help="EQ boost center frequency in Hz")
    enh.add_argument("--denoise", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "cover":
        from .pipeline import cover_pipeline
        cover_pipeline(args)
    elif args.command == "analyze":
        from .analyze import analyze_recording, diagnose_vocal_presence, format_report
        report = analyze_recording(args.input, segment_duration=args.duration)
        diagnosis = diagnose_vocal_presence(report)
        print(format_report(report, diagnosis))
    elif args.command == "separate":
        from .separate import separate_stems
        separate_stems(args.input, args.output_dir, args.model)
    elif args.command == "transcribe":
        from .transcribe import transcribe_audio
        transcribe_audio(args.input, args.model, args.compare)
    elif args.command == "enhance":
        from .enhance import enhance_vocals
        enhance_vocals(args.input, args.output, args.volume, args.eq_freq, args.denoise)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
