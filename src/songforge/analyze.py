"""
Spectral analysis precheck — diagnose recordings before separation.

The Darmok Problem: when a phone mic is closer to the guitar than to the
voice, the guitar's body resonance (80-250 Hz) absorbs the vocal
fundamentals (80-300 Hz). Demucs then classifies the voice as silence
because the signal is below the noise floor.

This module analyzes the input recording BEFORE running Demucs, providing:
- Spectral profile (centroid, rolloff, flux, flatness)
- Per-band energy distribution (sub-bass, bass, low-mid, mid, high-mid, treble)
- Vocal presence estimate based on formant-region energy
- SNR estimate for the vocal band
- A diagnostic report that recommends whether separation is likely to succeed

Usage:
    from songforge.analyze import analyze_recording, diagnose_vocal_presence
    report = analyze_recording("song.mp3")
    diagnosis = diagnose_vocal_presence(report)
    if diagnosis["recommendation"] == "skip_separation":
        print(diagnosis["warning"])
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Frequency band definitions (Hz) ─────────────────────────────────────────
# Standard psychoacoustic bands adapted for vocal/guitar analysis
BANDS = {
    "sub_bass":      (20, 60),      # Feel, not hear
    "bass":          (60, 250),     # Guitar body resonance lives here
    "low_mid":       (250, 500),    # Vocal fundamentals (male lower range)
    "mid":           (500, 2000),   # Vocal core, primary speech intelligibility
    "high_mid":      (2000, 4000),  # Vocal presence, formants
    "treble":        (4000, 8000),  # Consonants, air, brightness
    "air":           (8000, 16000), # Breath, sparkle
}

# Vocal-relevant bands for presence detection
VOCAL_BANDS = ["mid", "high_mid"]
VOCAL_FREQ_RANGE = (300, 4000)  # Conservative vocal detection window


@dataclass
class BandEnergy:
    """Energy measurement for a single frequency band."""
    name: str
    freq_low: float
    freq_high: float
    rms: float
    peak_db: float
    relative_energy: float  # proportion of total spectrum energy


@dataclass
class SpectralReport:
    """Full spectral analysis of a recording."""
    file: str
    duration_sec: float
    sample_rate: int
    channels: int
    spectral_centroid_hz: float      # Brightness center
    spectral_rolloff_85_hz: float    # Frequency below which 85% of energy sits
    spectral_flatness: float         # 0=noise-like, 1=tonal
    spectral_flux: float             # Average frame-to-frame change
    bands: list[BandEnergy] = field(default_factory=list)
    estimated_key: Optional[str] = None
    vocal_band_rms: float = 0.0
    instrumental_band_rms: float = 0.0
    vocal_to_instrumental_ratio_db: float = 0.0


def _run_ffprobe(input_file: str) -> dict:
    """Get file metadata via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        input_file
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    return json.loads(result.stdout)


def _extract_segment(input_file: str, duration: float = 30.0) -> str:
    """Extract a representative segment for analysis (first N seconds)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    
    # Convert to mono, 16-bit PCM at 44.1kHz for consistent analysis
    result = subprocess.run([
        "ffmpeg", "-y", "-i", input_file,
        "-t", str(duration),
        "-ac", "1",                    # mono
        "-ar", "44100",                # 44.1kHz
        "-sample_fmt", "s16",          # 16-bit
        "-vn",                         # no video
        tmp.name
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed: {result.stderr}")
    
    return tmp.name


def _compute_band_energies(wav_file: str) -> list[BandEnergy]:
    """Compute per-band RMS energy using ffmpeg astats filter.
    
    We use a series of bandpass filters to isolate each frequency range
    and measure its RMS level.
    """
    bands = []
    
    for name, (low, high) in BANDS.items():
        # Use ffmpeg's bandpass filter to isolate this band
        result = subprocess.run([
            "ffmpeg", "-y", "-i", wav_file,
            "-af", f"bandpass=f={(low+high)/2}:width_type=h:w={(high-low)/2},"
                   f"volumedetect",
            "-f", "null", "-"
        ], capture_output=True, text=True)
        
        # Parse mean_volume from stderr
        rms = 0.0
        peak_db = -99.0
        for line in result.stderr.split('\n'):
            if "mean_volume" in line:
                try:
                    rms = float(line.split("mean_volume:")[1].strip().replace(" dB", ""))
                except (ValueError, IndexError):
                    pass
            if "max_volume" in line:
                try:
                    peak_db = float(line.split("max_volume:")[1].strip().replace(" dB", ""))
                except (ValueError, IndexError):
                    pass
        
        bands.append(BandEnergy(
            name=name,
            freq_low=low,
            freq_high=high,
            rms=rms,
            peak_db=peak_db,
            relative_energy=0.0,  # computed after all bands collected
        ))
    
    # Compute relative energies
    # Convert dB to linear for proportional calculation
    import math
    linear_energies = []
    for b in bands:
        if b.rms > -90:  # above floor
            linear_energies.append(10 ** (b.rms / 20))
        else:
            linear_energies.append(0.0)
    
    total = sum(linear_energies) or 1.0
    for b, lin in zip(bands, linear_energies):
        b.relative_energy = round(lin / total, 4)
    
    return bands


def _compute_spectral_centroid(wav_file: str) -> tuple[float, float, float, float]:
    """Compute spectral centroid, rolloff, flatness, and flux from audio.

    These are measured directly from the samples via a short-time Fourier
    transform (numpy FFT over Hann-windowed frames). ffmpeg's astats filter
    is time-domain only and never emits these values — parsing its output
    for them silently returned zeros, so we compute them ourselves.

    Returns:
        (centroid_hz, rolloff_85_hz, flatness, flux) — zeros for empty or
        silent audio rather than raising, so the report can still render.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return (0.0, 0.0, 0.0, 0.0)

    try:
        data, sr = sf.read(wav_file, dtype="float32", always_2d=True)
        samples = data[:, 0]  # mono extraction already happened, take first channel
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)

    if len(samples) < 1024:
        return (0.0, 0.0, 0.0, 0.0)

    # Short-time Fourier transform: Hann-windowed frames, 50% overlap.
    frame = 1024
    hop = frame // 2
    win = np.hanning(frame)
    n_frames = 1 + (len(samples) - frame) // hop
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)

    mags = np.empty((n_frames, len(freqs)), dtype=np.float64)
    for i in range(n_frames):
        seg = samples[i * hop:i * hop + frame] * win
        mags[i] = np.abs(np.fft.rfft(seg))

    # Mean magnitude spectrum across frames (perceptual smoothing)
    spec = mags.mean(axis=0)
    total_energy = spec.sum()
    if total_energy <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    # Centroid: energy-weighted mean frequency
    centroid = float((spec * freqs).sum() / total_energy)

    # Rolloff: frequency below which 85% of spectral energy sits
    cumulative = np.cumsum(spec)
    idx = int(np.searchsorted(cumulative, 0.85 * total_energy))
    rolloff = float(freqs[min(idx, len(freqs) - 1)])

    # Flatness: geometric mean / arithmetic mean of magnitudes (0..1,
    # 0 = noise-like, 1 = tonal). Add epsilon to keep silent bins from
    # zeroing the geometric mean.
    eps = 1e-12
    geom = np.exp(np.mean(np.log(spec + eps)))
    arith = np.mean(spec + eps)
    flatness = float(geom / arith) if arith > 0 else 0.0

    # Flux: mean frame-to-frame L2 change in the magnitude spectrum
    if n_frames > 1:
        flux = float(np.linalg.norm(np.diff(mags, axis=0)) / (n_frames - 1))
    else:
        flux = 0.0

    return (round(centroid, 1), round(rolloff, 1), round(flatness, 4), round(flux, 4))


def analyze_recording(input_file: str, segment_duration: float = 30.0) -> SpectralReport:
    """Run full spectral analysis on a recording.
    
    Args:
        input_file: Path to the audio file.
        segment_duration: Seconds of audio to analyze (from start).
    
    Returns:
        SpectralReport with all measurements.
    
    Raises:
        RuntimeError: If ffmpeg/ffprobe are unavailable or analysis fails.
        FileNotFoundError: If the input file doesn't exist.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Get file metadata
    probe = _run_ffprobe(input_file)
    format_info = probe.get("format", {})
    streams = probe.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    
    duration = float(format_info.get("duration", 0))
    sample_rate = int(audio_stream.get("sample_rate", 44100))
    channels = int(audio_stream.get("channels", 1))
    
    # Extract analysis segment
    seg_file = _extract_segment(input_file, min(segment_duration, duration or segment_duration))
    
    try:
        # Compute band energies
        bands = _compute_band_energies(seg_file)
        
        # Compute spectral features
        centroid, rolloff, flatness, flux = _compute_spectral_centroid(seg_file)
        
        # Aggregate vocal vs instrumental band energy
        vocal_rms = min(
            next((b.rms for b in bands if b.name == "mid"), -60),
            next((b.rms for b in bands if b.name == "high_mid"), -60)
        )
        instrumental_rms = next((b.rms for b in bands if b.name == "bass"), -20)
        
        # Ratio in dB (positive = vocals dominate, negative = instruments dominate)
        ratio = vocal_rms - instrumental_rms if vocal_rms > -90 and instrumental_rms > -90 else -99
        
        report = SpectralReport(
            file=input_file,
            duration_sec=duration,
            sample_rate=sample_rate,
            channels=channels,
            spectral_centroid_hz=centroid,
            spectral_rolloff_85_hz=rolloff,
            spectral_flatness=flatness,
            spectral_flux=flux,
            bands=bands,
            vocal_band_rms=vocal_rms,
            instrumental_band_rms=instrumental_rms,
            vocal_to_instrumental_ratio_db=round(ratio, 2),
        )
        
        return report
        
    finally:
        # Clean up temp file
        Path(seg_file).unlink(missing_ok=True)


def diagnose_vocal_presence(report: SpectralReport) -> dict:
    """Diagnose whether vocals are likely recoverable from this recording.
    
    This implements the lesson from the Darmok incident: when the vocal band
    RMS is more than 15 dB below the instrumental band, Demucs will likely
    classify the vocals as silence.
    
    Returns:
        Dict with keys:
        - recommendation: "proceed" | "caution" | "skip_separation"
        - confidence: float (0-1)
        - warning: str (human-readable diagnosis)
        - vocal_band_rms_db: float
        - instrumental_band_rms_db: float
        - ratio_db: float
        - dominant_band: str
    """
    ratio = report.vocal_to_instrumental_ratio_db
    
    # Find dominant band
    dominant = max(report.bands, key=lambda b: b.relative_energy)
    
    # Classify vocal presence
    # Based on Darmok data: vocal RMS was ~-68 dB vs instrumental ~-20 dB = -48 dB ratio
    # Anything worse than -15 dB is problematic for Demucs
    if ratio > -5:
        recommendation = "proceed"
        confidence = 0.9
        warning = ""
    elif ratio > -15:
        recommendation = "caution"
        confidence = 0.5
        warning = (
            f"Vocal band is {abs(ratio):.1f} dB below instrumental. "
            "Separation may partially succeed but vocal quality will be degraded. "
            "Consider providing known lyrics for lyric-matched generation."
        )
    else:
        recommendation = "skip_separation"
        confidence = 0.85
        warning = (
            f"VOCAL BELOW NOISE FLOOR: vocal band is {abs(ratio):.1f} dB below "
            f"instrumental band. Dominant energy is in '{dominant.name}' "
            f"({dominant.freq_low}-{dominant.freq_high} Hz). "
            "Demucs will likely classify vocals as silence. "
            "RECOMMENDATION: Skip stem separation. Provide known lyrics directly "
            "and use lyric-matched generation instead."
        )
    
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "warning": warning,
        "vocal_band_rms_db": round(report.vocal_band_rms, 2),
        "instrumental_band_rms_db": round(report.instrumental_band_rms, 2),
        "ratio_db": ratio,
        "dominant_band": dominant.name,
    }


def format_report(report: SpectralReport, diagnosis: dict) -> str:
    """Format the spectral report and diagnosis for terminal output."""
    lines = [
        "═" * 60,
        "SongForge — Spectral Analysis Precheck",
        "═" * 60,
        "",
        f"  File:            {report.file}",
        f"  Duration:        {report.duration_sec:.1f}s",
        f"  Sample rate:     {report.sample_rate} Hz",
        f"  Channels:        {report.channels}",
        "",
        "─── Frequency Band Energy ───────────────────────────────",
    ]
    
    for band in report.bands:
        bar_len = int(band.relative_energy * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        lines.append(
            f"  {band.name:<12} {band.freq_low:>5}-{band.freq_high:<5} Hz "
            f"│{bar}│ {band.rms:>6.1f} dB ({band.relative_energy:.1%})"
        )
    
    lines.extend([
        "",
        "─── Vocal Presence Diagnosis ────────────────────────────",
        f"  Vocal band RMS:       {report.vocal_band_rms:.1f} dB",
        f"  Instrumental band RMS: {report.instrumental_band_rms:.1f} dB",
        f"  V/I ratio:            {report.vocal_to_instrumental_ratio_db:.1f} dB",
        f"  Dominant band:        {diagnosis['dominant_band']}",
        "",
    ])
    
    # Color-code the recommendation
    rec = diagnosis["recommendation"]
    icon = {"proceed": "✅", "caution": "⚠️", "skip_separation": "🚫"}.get(rec, "?")
    lines.append(f"  {icon} RECOMMENDATION: {rec.upper()}")
    if diagnosis["warning"]:
        lines.append(f"  {diagnosis['warning']}")
    lines.append("")
    lines.append("═" * 60)
    
    return "\n".join(lines)


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    """Standalone CLI: python -m songforge.analyze song.mp3"""
    if len(sys.argv) < 2:
        print("Usage: python -m songforge.analyze <audio_file>")
        sys.exit(1)
    
    report = analyze_recording(sys.argv[1])
    diagnosis = diagnose_vocal_presence(report)
    print(format_report(report, diagnosis))


if __name__ == "__main__":
    main()
