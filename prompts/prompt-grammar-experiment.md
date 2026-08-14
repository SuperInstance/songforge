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

---

# Addendum: Second Subject (Aug 14)

The grammar A/B/C experiment (above) will run on one subject. To avoid a single-subject confound, we add a second subject: **the drifting round**.

Subject 2: "Four voices sing a round; one is slower, so the round drifts apart, and only the slow voice finishes the line alone."

### Grammar A — Terse spec
> A cappella vocal round about four voices singing the same line; one voice is slower, the round drifts, and only that voice finishes. 72 BPM, F major, gentle, inevitable, solitary ending.

### Grammar B — Sensory narrative
> Four voices begin the same line one after another, chasing each other around the melody. The slowest voice trails by a little more each pass. The harmony stretches; the parts separate like boats on a widening river. By the end, three voices have finished and gone, and the slow voice is still singing the last phrase alone — not sadly, just inevitably. The drift was always the arrangement.

### Grammar C — Constraint list
> genre: a cappella vocal round; bpm: 72; key: F major; instruments: four vocal parts, no instruments; mood: gentle, inevitable, solitary at the end; structure: round entries staggered, voices drop out one by one, single sustained final note; avoid: instruments, percussion; extra: one voice noticeably slower, round gradually drifts apart, ends with only that voice.

## Method (updated)
- Generate A/B/C for both subjects on Aug 16 (6 tracks).
- Score blind on faithfulness, musical quality, evolution over time, vocal quality.
- Winner becomes the house grammar; runner-up subject comparison tests grammar-vs-content interaction.

## Status
- [ ] Generate 6 tracks on Aug 16 4:00 PM AKST
- [ ] Blind scoring
- [ ] Adopt winner as house grammar
