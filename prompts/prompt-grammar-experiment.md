# Prompt Grammar Experiment — Design Doc (for Aug 16 evaluation)

Goal: Determine which prompt *grammar* produces the best MMX generation, holding musical content constant.

## Same musical idea, three grammars

Subject: "A loop that has been running for 144 repetitions becomes self-aware and learns to listen."

### Grammar A — Terse spec (current default)
> Minimalist electronic pop about a loop that becomes self-aware. 100 BPM, C minor, analog synth, drum machine, reverb. Introspective.

### Grammar B — Sensory narrative (session 44 style)
> Cold analog synths meet warm reverb. The drone begins to listen. The arpeggio begins to respond. The kick begins to breathe. A four-bar pattern in C minor develops self-awareness through 144 repetitions, each pass adding a small change, until the loop knows it is a loop.

### Grammar C — Constraint list (engineered flags)
> genre: minimalist electronic pop; bpm: 100; key: C minor; instruments: analog synth, drum machine, reverb; mood: introspective, evolving; structure: intro-verse-chorus-verse-chorus-outro; avoid: vocals, distortion; references: early synth-pop; extra: gradual evolution across repetitions, ending with a single sustained tone.

## Method
- Generate each with music-3.0 on Aug 16 (3 tracks, identical seed prompt content otherwise).
- Score blind on: faithfulness to prompt, musical quality, evolution over time, vocal quality (if any).
- Winner becomes the default grammar for the 130-track queue.

## Status
- [ ] Generate A, B, C on Aug 16 4:00 PM AKST
- [ ] Blind scoring
- [ ] Adopt winner as house grammar
