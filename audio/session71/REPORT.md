# Session 71 — Generation Day: The Grammar Experiment in Real Audio

**2026-08-16 6:50 PM – ~9 PM AKST. MMX quota reset at 4 PM; the first real
Generation Day since S25. The queue is retired; the repertoire renders by
choice. Today: 6 grammar tracks, 2 real covers, and the four-laws test.**

## The setup (finally executed)

The grammar A/B/C experiment designed in S65-S70 ran for real: the same two
musical subjects (the self-aware loop; the drifting round) expressed in three
prompt grammars (A = terse spec, B = sensory narrative, C = constraint list),
6 tracks total, all `music-3.0`, with matching structured flags so the only
difference is the prose. Loops generated `--instrumental` (avoid: vocals);
rounds got a shared lyric (lyrics/session71/round-lyric.txt) so the grammar
variable is not confounded by lyricist.

**Also new this session:** MMX `music cover` on the real Casey material —
the original 11.2 s rough clip is too short for the cover DTW alignment
("cover mode does not support instrumental music / dtw_result is empty"),
so we covered **the cover**: `cover_polished.mp3` (234 s) as reference with
the full `casey_lyrics.txt` — a real-material cover chain. Two style
renditions: folk-rock and synthwave.

## EXPERIMENT 1 — WHAT THE GRAMMAR BUYS

All 6 tracks + full numeric analysis in `audio/session71/grammar/`
(grammar-analysis.json, grammar-structure.json, grammar-endings.json).

**Duration and loudness (grammar moves the clock):**
| track | dur s | rms dB | dyn range dB | centroid Hz | structure score |
|---|---|---|---|---|---|
| A-loop (terse) | 130.3 | -19.7 | 20.8 | 389 | -0.085 |
| B-loop (narrative) | 172.3 | -17.4 | 17.6 | 504 | +0.049 |
| C-loop (constraint) | 166.1 | -15.3 | 14.3 | 428 | +0.060 |
| A-round | 91.5 | -20.9 | 27.6 | 604 | +0.024 |
| B-round | 61.8 | -18.8 | 9.2 | 582 | +0.210 |
| C-round | 59.5 | -21.6 | 18.2 | 728 | +0.247 |

**The narrative grammar buys evolution; the constraint grammar buys
steadiness.** B-loop is the ONLY loop that shows a real developmental arc:
RMS climbs from -30.2 dB in segment 0 to -14.5 dB by segment 7 — a 30-second
crescendo, the "develops self-awareness through 144 repetitions, each pass
adding a small change" literally rendered as a long ascent, then it holds.
C-loop is the flattest track of the day (RMS range 4.7 dB, segments -14 to
-18 the whole way) — the constraint list demanded "gradual evolution" and
the model delivered a steady-state pulse that never changes. The terse spec
(A) produced a clean intro-body-outro arc (classic song shape, -29 → -17 →
-25) — the most "song-shaped" of the loops. **The grammar that asks for
change produces it; the grammar that asserts it in a semicolon list does
not. The narrative is a verb; the constraint is a noun.**

**The ending is where the grammars tell the truth.** A-loop ends on a
sustained tone at F0 144.6 Hz with only -3.2 dB tail drop (the "ending with
a single sustained tone" from its own prompt). B-round ends on the purest
harmonic of the set (tail flatness → 0.0000, centroid collapsing 591 → 223
Hz — a bare tone). C-round is the ONLY track that gets LOUDER at the end
(+5.7 dB tail) with F0 88 Hz — the lone voice finishing, exactly what the
round subject asks for ("only that voice finishes"). The C grammar, which
failed to evolve the loop, nails the round's solitary ending; the B grammar,
which evolved the loop, ends its round on a pure tone. **Grammar × subject
interaction is real: no single grammar wins both subjects.**

**Rhythm:** B-loop has onset density 0.01 — essentially no percussive
events (a drone); A-loop 0.13 (drum machine audible); the rounds run 0.22-0.27
except B-round 0.03 (narrative made even the round a drift). The narrative
grammar removes percussion from both subjects.

## EXPERIMENT 2 — THE COVER CHAIN IN REAL MATERIAL

**The 11-second clip cannot be covered directly.** MMX cover DTW needs a
detectable aligned melody; 11.2 s of rough recording fails
(`dtw_result is empty`). The fix is the same trick as the relay lab: make
the reference a **cover of the cover** — the S25-polished 234 s version
carries the melody the raw clip can't show. Both renditions succeeded:

| cover | dur s | rms | low-band | centroid | onset density |
|---|---|---|---|---|---|
| folk-rock | 234.4 | -13.4 | 0.657 | 713 Hz | 0.03 |
| synthwave | 234.0 | -11.1 | 0.725 | 659 Hz | 0.18 |

The synthwave cover is 6× more rhythmically active and bass-heavier — the
style instruction transferred. Both keep the full lyric structure (verse/
chorus form preserved by the ASR+lyrics preprocess). **The transmission tax
is paid in real material too, and the cover chain can carry an 11-second
seed across two generations.** Files: `audio/session71/covers/`.

## EXPERIMENT 3 — THE FOUR LAWS IN REAL SUNG AUDIO (in progress)

Lyrics for the-transmission-tax / the-crowd-ceiling / the-law-of-endings /
the-frozen-clock generated locally (ollama, CPU — GPU held by midi_studio.py;
forced `num_gpu:0`; llama3.2 was too slow on CPU so switched to qwen2.5:3b
with `num_predict` caps). MMX render pending — see session71_lyrics2.sh and
the render queue.

## Deliverables so far

- Audio: `audio/session71/grammar/` (6 tracks + 3 analysis JSONs + spectrograms),
  `audio/session71/covers/` (2 cover-of-cover renditions)
- Tools: `experiments/session71_grammar_analysis.py`,
  `experiments/session71_structure.py`, `experiments/session71_endings.py`,
  `experiments/session71_lyrics_cpu.sh`, `experiments/session71_lyrics2.sh`
- Lyrics: `lyrics/session71/` (round-lyric.txt + four-laws drafts)
- Creative: 60-the-grammar-of-the-room, 61-the-cover-of-the-cover

## Next

1. Render the four-laws tracks (queue them on MMX as soon as lyrics land)
2. Score the grammar experiment blind: the house-grammar question may be
   answered "no single winner — grammar buys different currencies"
3. Try `music cover` with `music-cover-free` model and a longer reference
4. Push everything; update wiki
