# One Day In E — Iteration 3: Deep Separation R&D

## Date: 2026-08-06
## Status: All automated separation exhausted. Re-recording is the answer.

## Separation Model Comparison

| Model | Vocal RMS (dB) | Vocal Peak (dB) | Notes |
|-------|----------------|-----------------|-------|
| htdemucs (default) | -74.0 | -50.4 | Original attempt |
| htdemucs_ft | -72.1 | -56.5 | Fine-tuned, slight improvement |
| htdemucs_ft (shifts=10, overlap=0.75) | -74.5 | -66.2 | More averaging = worse |
| mdx | -72.6 | -47.5 | Different architecture |
| **mdx_extra** | **-68.5** | **-42.5** | **BEST: 5.5 dB improvement** |
| hdemucs_mmi | -74.5 | -54.9 | No improvement |
| htdemucs_6s | -75.3 | -59.9 | 6-stem separation |

**Conclusion:** mdx_extra is the best model, but even at -68.5 dB the vocals are ~50 dB below a normal recording. All models classify this as instrumental.

## Spectral Editing Results

| Technique | Result |
|-----------|--------|
| Bandpass 300-3000 Hz | RMS -27 dB (guitar + vocals fused) |
| Narrow 500-2000 Hz x10 boost | RMS -21 dB (still guitar) |
| Spectral gating | No improvement, added artifacts |
| Soft masking (6s instrumental subtraction) | 0.5% voiced frames, C3 only |
| Direct subtraction (original - no_vocals) | Artifacts, not vocals |

## Whisper Transcription Results

| Input | Model | Result |
|-------|-------|--------|
| Original | base | "" (nothing) |
| Bandpass boosted | base | "Music" |
| Narrow boosted | base | "Music" |
| Spectral gated | base | "Music" |
| mdx_extra vocals x100 | base | "Huh?" (low confidence) |
| Residual boosted | base | "Thank you very much" (hallucination) |
| Residual boosted | small | "Thank you." (hallucination) |
| Subtraction residual | medium | "This video was made possible..." (hallucination) |

**Conclusion:** Whisper cannot detect any speech in any variant. The vocals are below detection threshold across all preprocessing methods.

## Melody Extraction

- pYIN on original: C2 (65 Hz) — guitar body resonance
- pYIN on bandpass: E4, F4, G#4 — could be vocal harmonics or guitar overtones, inconclusive
- pYIN on soft-masked: 0.5% voiced, C3 only — no usable melody contour
- basic-pitch: failed to install (Python 3.14 incompatibility)

## Song Analysis (from 6s model)

- **Key:** E major (r=0.782) / G#m (r=0.608)
- **Tempo:** ~110 BPM
- **Duration:** 11.2 seconds
- **Clean backing track:** `clean_backing_track.wav` (guitar + bass from 6s model)

## Failed Installations

- AudioSR: Python 3.14 incompatible (numpy build fails)
- audio-separator (BS-Roformer): Python 3.14 incompatible (diffq build fails)
- basic-pitch: Python 3.14 incompatible (numpy build fails)
- Python 3.11 venv: OOM killed during torch installation

## Recommendation

**Re-record.** A 30-second phone recording at 6 inches from the mouth will produce vocals at -20 to -10 dB — 10,000x improvement over current. See `RECORDING_GUIDE.md`.

## Tools That Might Work (If We Had Compatible Python)

- **BS-Roformer** (Band-Split RoPE Transformer): Current SOTA for vocal separation
- **UVR5** (Ultimate Vocal Remover): GUI/toolchain with multiple model support
- **AudioSR**: Diffusion-based audio super-resolution
- **LALAL.AI Phoenix/Orion**: Cloud-based separation (paid)

These are designed for recordings where vocals are present but messy — not recordings where vocals are 50+ dB below the noise floor. Even SOTA tools likely cannot recover this recording.
