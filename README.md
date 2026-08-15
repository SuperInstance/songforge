# SongForge

> AI song covers from rough recordings, plus a research lab for transmission-chain experiments.

SongForge takes an old, imperfect recording of an original song and turns it
into a modern AI cover: separate the vocals, verify the melody, polish the
voice, generate a new performance, mix it back over the original instruments.
It also hosts a running experiment series — the **relay rounds** — that
studies what happens when a song is handed from voice to voice through a
chain of models, and measures the signal's fate in decibels.

**You'll need the song's lyrics.** SongForge verifies transcription against
them (`--compare`); the cover step takes them as input. It's a verification
tool as much as a generator.

## The Problem

You have an old recording. The vocals are buried, the mix is rough, and modern
AI cover tools can't even detect the melody in it. SongForge bridges the gap:

1. **Separates** vocals from instruments (Demucs source separation)
2. **Transcribes** the isolated vocals (Whisper) to verify against known lyrics
3. **Enhances** the vocal track (volume, EQ, optional de-noise)
4. **Generates** a new cover via AI music generation (MMX / MiniMax)
5. **Mixes** the cover with the original instrumental for a polished result

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

# Compare transcription against known lyrics (cover verification)
python -m songforge transcribe --input vocals.wav --compare lyrics.txt

# Spectral precheck — diagnose a recording BEFORE separation
python -m songforge analyze --input song.mp3

# Enhance vocal quality (volume, EQ center freq, optional de-noise)
python -m songforge enhance --input vocals.wav --output enhanced.wav \
  --volume 3.0 --eq-freq 2000 --denoise
```

(`--eq-freq` is the peaking-EQ center frequency in Hz, not a band selector.)

## Architecture

```
src/songforge/
  cli.py        — command dispatch (cover | separate | transcribe | analyze | enhance)
  pipeline.py   — the full 5-stage cover pipeline; owns _generate_cover / _mix_tracks
  separate.py   — Demucs stem separation
  transcribe.py — Whisper transcription + optional --compare against known lyrics
  analyze.py    — spectral precheck: measures spectral features for real
                  (RMS profile, edges, gaps, spectral centroid), no astats parsing
  enhance.py    — vocal polish: volume, peaking EQ, optional de-noise
```

Package layout: setuptools `src/` layout, console script `songforge`,
Python 3.10+, `ffmpeg` on PATH.

**Hardware:** Demucs and Whisper are heavy. Tested on GPU machines; CPU-only
runs work but separation/transcription are slow. The experiment tools
(`experiments/`) are light — they only mix and analyze wav files.

## The Relay Lab (`experiments/`)

The experiments, not the covers, are the soul of this repo. Four TTS voices
(lessac, norman, joe, amy) sing one line each, handing off via equal-power
crossfades of duration **X** seconds. Vary X and you change how much of each
voice's *edges* survive into the next. Three measurable quantities, tracked
across sessions 63-66:

- **Conservation** — total signal energy, relay vs. a "crowd" control where
  all voices sing at once. Does the handoff lose energy?
- **Tax** — chain energy ÷ crowd energy at the same material. The price of
  being a chain instead of a crowd.
- **Resonance** — periodic dips/teeth in the energy and loudness-profile of
  the relay as X sweeps, tied to the cast's duration differences.

| Tool | What it does | Run it |
|------|--------------|--------|
| `build_relay.py` | Build a relay round (chain) or staircase (crowd control) from a voice dir (must contain `lessac/norman/joe/amy.wav`); `build_relay_vx()` supports per-handoff crossfade widths | `python3 experiments/build_relay.py <voice_dir> <out_dir> [--x 1.0]` |
| `depth3_relay.py` | S66 build: relay-of-relays-of-relays (depth 3) + depth-2 resonance sweep (X 0→5.5 s step 0.25) | `python3 experiments/depth3_relay.py <session65_dir> <out_root>` |
| `analyze_depth3.py` | S66 analysis: tax series across depths 1→3, resonance teeth (std/energy/silence), pause-structure prediction test | `python3 experiments/analyze_depth3.py <s64_dir> <s65_dir> <s66_dir>` |
| `refine_resonance.py` | S66 refinement: FFT period of the std-vs-X curve (both depths) + envelope-correlation prediction replacing the failed pause test | `python3 experiments/refine_resonance.py <s64_dir> <s65_dir> <s66_dir>` |
| `session67_build.py` | S67 build: depth-4 relays + frozen-clock targeted sweep + chosen-X (analyzer-as-composer) relays | `python3 experiments/session67_build.py <s66_dir> <out_dir>` |
| `session67_analyze.py` | S67 analysis: depth-4 tax/crowd/fairness + frozen-clock teeth + composer placement test | `python3 experiments/session67_analyze.py <s64> <s65> <s66> <s67>` |
| `session68_census.py` | S68 build+analyze: N=8 census (fwd/rev/interval-5) + f32 no-clip twin — veq → N exactly, container tax curve | `python3 experiments/session68_census.py <s66_dir> <out_dir>` |
| `session68_composer.py` | S68 v2 aiming loop (gain 2, ±2 s argmin measurement) — the loop rings, chases ghosts | `python3 experiments/session68_composer.py <s66_dir> <out_dir>` |
| `session68_aim3.py` | S68 v3 damped aimer (gain 0.5, predicted-center measurement, ghost reporting) + medium's ceiling curve (N=2/4/8, s16 vs f32) | `python3 experiments/session68_aim3.py <s66_dir> <out_dir>` |
| `morph_sweep.py` | S65: sweep X over the relay, watch the morph → tax curve with resonance teeth | see file docstring |
| `analyze_conservation.py` | Shared wav reader + windowed power profiles; the analysis workhorse | imported by the others |
| `generate_lyrics.sh` | Generate lyrics from a local ollama model at temperature T (S64 path-fix: reads prompt files instead of their paths) | `./generate_lyrics.sh <model> <temp> <outfile> [prompt]` |

Audio corpus lives in `audio/sessionNN/` (gitignored — regenerable with the
build tools). Readable artifacts — the lyric transcripts — live in
`lyrics/sessionNN/` and are committed.

## Session 68 findings (2026-08-15, morning)

The census is exact, the aimer rings (then converges), and the medium has a
ceiling curve.

- **veq → N exactly, in float32.** N=2/4/8 staircases of the same layer-3
  cast: veq 2.010 / 4.025 / **8.066** — the census law is arithmetic. The
  0.5–0.8% overshoot is the self-correlation bonus (identical material added
  to itself correlates; cross-terms are positive).
- **The medium's ceiling is a curve, monotone in N.** Container tax (s16
  energy ÷ f32 energy): 0.9976 → 0.9920 → **0.9738**. The s16 container
  takes 2.6% of an 8-voice census vs 0.24% at N=2. The "ceiling is the cast"
  (S67) needs its amendment: the cast sets the census, the container levies
  the tax, and the tax grows with the crowd.
- **The census fate is the frozen clock exiting.** N=8 fairness delta 1.45 dB
  is NOT entry order (both orders negative; a census enters everyone at
  once). The 10% tail window catches the exit ramp, staggered by cast
  durations — the two census files differ by exactly the 2.1 s birthmark
  (250.39 vs 252.49 s). The clock walks out through the exit ramp.
- **The aimer rings at gain 2.** v2 loop: +0.69 → −1.62 → −0.47 s, a limit
  cycle pinned against the X=5.5 wall; the ±2 s argmin sometimes measures
  the inherited fade-rim ghost (72.8 dB) instead of the placed seam.
- **The damped aimer converges.** v3 (gain 0.5, measure at the predicted
  center anchor − X/2, report the ghost separately): handoff 0 lands at a
  −65 ms fixed point in one round, 12–15 dB deep. Handoff 2 stays shallow
  (≈0 dB): position is free, depth is rented — the material lends only what
  its correlation at that width allows.
- **The pre-screen floor was wrong.** v2 scanned from X=0.30 and saturated
  there; v3's true floor is X=0.25 for all three handoffs (max|corr| 0.85–0.88).

## Session 67 findings (2026-08-14, evening)

Depth 4 (256 voices, ~960 s), the frozen clock, and the analyzer-as-composer.

- **The tax holds at one.** Depth-4 tax: 0.9999 / 0.9988 / 0.9962
  (X = 0.3 / 1.0 / 2.0). Deficit compounding factor ≈ 0.29 (X=2.0),
  matching the ~0.25/layer prediction.
- **The crowd ceiling is refuted — the ceiling is the cast.** Crowd veq
  series 2.02 → 3.14 → 3.40 → **3.99**: a staircase of N long voices
  approaches N·E_single; the "saturation" was the entry-ramp fraction.
  Togetherness is bounded by a census.
- **The fate amortizes, it is not a fixed point.** With proportional
  (10%-of-file) tail windows the entry-order delta runs 3.23 → 0.73 →
  0.07 → 0.10 dB across depths 1→4. The S64–S66 half-decibel "fate" was
  a fixed-window artifact. Twin asymptotes: tax → 1, fate → 0.
- **The frozen clock.** A composed file lasts Σdur − 3X, so cast duration
  differences (2.1 / 3.0 / 5.1 s) survive every recursion unchanged — the
  internal clock is made of crossfades. Depth-4 sweep rings at the depth-2
  addresses (dip 3.25, std tooth 3.25 / 5.25).
- **The analyzer composes.** Per-handoff chosen-X relays: correlation
  minima place handoff dips 13–17 dB below body; maxima place humps
  (+1.5/+2.6 dB); the zero-correlation "control" lands at −41.6 dB —
  zero is not neutral, it is empty. The placed silences are manufactured:
  composed files carry inherited fade-rims (−81 to −92 dB tails) the raw
  voices never had.

## Session 66 findings (2026-08-14)

The depth-3 relay run answered the open question from S65: **does the
transmission tax asymptote?**

- **The tax asymptote is 1.0 — the interior is tax-free.** Chain energy
  relative to the crowd of the same material, at X=1.0: depth 1 → 0.735,
  depth 2 → 0.948, depth 3 → 0.998. The deficit compounds toward zero:
  0.265 → 0.052 → 0.002. Layer 1 paid full price on raw voice edges; every
  deeper layer pays only the rounding tax. (The S65 guess of an 0.85
  asymptote was wrong — it's 1.0.)
- **The crowd has a ceiling.** Crowd-of-chains-of-chains hit **3.40 veq**
  (voice-equivalents — normalized loudness relative to a single voice; the
  densest ever measured), but the increments are collapsing: 1.12 → 0.26.
  Past four voices, more buys almost nothing — the room is full.
- **Fairness is a fixed point.** Entry order → ending level delta ≈ **0.53 dB
  at depth 3** — the same half-decibel of fate as depth 1. Order doesn't
  matter, at any depth.
- **Resonance period corrected.** The fine sweep (0.05 s steps) showed teeth
  at 0.25 s spacing — the true period is the *half*-difference of the largest
  cast duration difference, not the full difference. Depth-2 confirmed via
  FFT: dominant period 2.875 s vs adjacent-difference prediction 3.0 s
  (cast diffs 2.1 / 3.0 / 5.1 s).
- **Pause-structure prediction refuted.** The idea that resonance teeth come
  from aligned internal silences at handoff overlaps predicted *no* teeth
  (the piper voices have almost no internal pauses at 250 ms resolution).
  The envelope-correlation model — energy envelopes of tail vs. head lining
  up as X varies — produces the teeth instead.

## The Prompt Queue

Prompts live in `prompts/*.json` — one file per song design, each a full
spec: name, prompt text, genre, instruments, BPM, key, mood. Examples:
`conservation-of-signal.json`, `the-transmission-tax.json`,
`the-crowd-ceiling.json`, `the-fixed-point.json`, `the-frozen-clock.json`,
`the-forgiving-machine.json`, `the-manufactured-silence.json`,
`the-ceiling-is-the-cast.json`, `the-census.json`, `the-aiming-composer.json`,
`the-law-of-endings.json`, `the-mediums-ceiling.json`, `the-damped-aimer.json`,
`the-exit-ramp.json`. (42 designs, committed.)
`prompts/prompt-grammar-experiment.md` documents the grammar study that
shapes how prompts are written.

The **track queue** is the fleet-level roster of songs to generate on the
next Generation Day — a scheduled batch run (e.g. Aug 16 2026, 4 PM AKST).
"Fleet" here means the multi-agent writing/music system this repo belongs
to: songforge designs prompts, ai-writings holds the generated corpus
(`index.json`, 8,485 pieces), and the queue count (154 tracks) is the
manifest of what Generation Day will produce. The queue itself is a roster,
not a file in this repo — the repo holds the *designs*, the fleet holds the
*roster*.

## Tests

```bash
python -m pytest          # 109 passing (as of 2026-08-14)
python -m pytest -v       # verbose
```

Six modules: CLI dispatch, pipeline orchestration, spectral analysis
(incl. edge cases), enhancement, separation, transcription. CI runs on
GitHub Actions (push + PR) on Python 3.12.

## Known limitations

- Covers need the original lyrics — SongForge verifies, it doesn't invent.
- The generation stage depends on the MMX CLI; without it, everything up to
  `enhance` still works.
- A source too degraded (no detectable melody, vocals fully buried) will
  produce a cover that inherits the damage. `analyze` tells you before you
  commit to the pipeline.
- The relay experiments use synthetic piper voices (no real singing
  artifacts) — findings describe the model chain, not human singers.

## Requirements

- Python 3.10+
- ffmpeg on PATH
- Demucs, Whisper, librosa, pydub, soundfile (auto-installed via requirements)
- MMX CLI (for cover generation)

## License

MIT
