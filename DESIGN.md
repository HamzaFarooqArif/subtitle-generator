# Offline GPU Subtitle Generation Pipeline — Design

Target hardware: **NVIDIA RTX 3070, 8 GB VRAM**, Windows 11, Python 3.11.
Constraint: **fully local**, no cloud APIs. All models downloaded once, then `HF_HUB_OFFLINE=1`.

## Target material

**Unscripted consumer recordings**, not broadcast or studio audio. Nearly every
decision below follows from this, so it is worth stating precisely:

- **Single built-in microphone.** No centre channel to isolate, no boom, no gain
  control. Speaker-to-mic distance varies continuously.
- **Wide dynamic range within one file** — near-silence seconds after a shout,
  which is what rules out integrated loudness normalisation. See §4.3.
- **Speech that is genuinely ambiguous**, sometimes with no single correct
  transcript. The pipeline's job on such passages is to *decline*, not to guess.
- **Low-energy and unvoiced speech.** Under-represented in Whisper's training
  data and largely unvoiced, which degrades the acoustic model and the
  pitch-dependent parts of VAD and forced alignment alike.
- **Non-speech vocalization and room noise.** The single most dangerous input for
  Whisper: it does not answer with silence, it answers with fluent invented text.
  This is why §4.6 exists at all.
- **Mid-file language switching**, including intra-sentence code-switching. The
  window-level LID path in §4.4 is therefore mandatory, not optional.

**Honest expectation setting.** On genuinely unclear speech no model available
today — local or cloud — will produce a clean transcript. The realistic goal is
a good first draft plus *accurate confidence signals*, so that human review time
goes to the 10% of the file that needs it instead of being spread evenly. A
pipeline that silently emits confident nonsense is worse than one that emits
less and flags what it skipped. Every design choice below follows from that.

**Privacy.** Personal footage never leaves the machine. `enforce_offline()`
sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and disables hub telemetry at the
start of every run, and `sgen models verify` proves the model paths resolve with
the network disabled. The only command permitted to use the network is
`sgen models pull`.

---

## 1. Recommended stack (the short answer)

| Stage | Tool | Model / setting |
|---|---|---|
| Demux + resample | **ffmpeg / ffprobe** | 16 kHz mono `pcm_s16le`, dialogue-aware channel selection |
| Speech enhancement (conditional) | **Demucs** (music beds), **DeepFilterNet3** (stationary noise) | `htdemucs` 2-stem; DFN3 default |
| VAD | **Silero VAD v5** (bundled in faster-whisper) | 250 ms min speech, 400–700 ms min silence |
| Language ID | faster-whisper multi-segment LID; **SpeechBrain VoxLingua107 ECAPA** for code-switching | 8–12 chunks sampled across file |
| ASR (primary) | **faster-whisper ≥ 1.1** (CTranslate2) `BatchedInferencePipeline` | `large-v3`, **`compute_type="float16"`**, `batch_size=8–16` |
| ASR (fast tier) | same runtime | `large-v3-turbo` CT2, `float16` |
| ASR (fallback for 8 GB pressure) | same runtime | `large-v3`, `int8_float16` |
| **Non-speech / hallucination gating** | own module, [sgen/gating.py](sgen/gating.py) | confidence + repetition + word-rate + phrase blacklist |
| Forced alignment | **WhisperX align module** / `torchaudio.functional.forced_align` | wav2vec2 CTC per language |
| Diarization (optional) | **pyannote/speaker-diarization-3.1** | gated download, then offline |
| Cue building | own module + **pysubs2** | CPS/line-length/shot-change aware |
| Output | `.srt`, `.vtt`, `.ass`, plus `.sgen.json` sidecar | sidecar is the source of truth |
| Post-editing | **Subtitle Edit** (Windows, free) | waveform + CPS column + shot changes |

**Headline choice:** `faster-whisper large-v3` in **float16** with **VAD-chunked batched inference**, then **wav2vec2 forced alignment**. float16 (not int8) is the right default on a 3070 — the weights are ~3.1 GB, so you have room, and int8 quantization of Whisper measurably hurts on exactly the audio you care about (accents, noise, low SNR). Use int8_float16 only if you need to co-resident other models or push `batch_size` higher.

---

## 2. Why these choices

**faster-whisper over openai-whisper / whisper.cpp.** CTranslate2 gives ~4× the throughput of the reference PyTorch implementation at lower VRAM, exposes every decoding knob (temperature fallback, repetition penalty, `hotwords`), and supports batched VAD-chunked inference natively since 1.1. `whisper.cpp` is excellent for CPU/low-VRAM but you'd be leaving the 3070 idle.

**faster-whisper directly rather than WhisperX end-to-end.** WhisperX pioneered the VAD-chunk + batch + forced-align recipe, but it pins dependencies aggressively and its ASR stage is now a thin wrapper over what faster-whisper does natively. Depend on `faster-whisper` for ASR and borrow WhisperX only for `load_align_model` / `align` (or call `torchaudio.functional.forced_align` yourself). You keep control of batching, memory, and the retry logic. WhisperX as a single package remains the reasonable "get it working tonight" option.

**large-v3 over turbo as the accuracy default.** `large-v3-turbo` (809 M params, 4 decoder layers) is 3–5× faster and close on clean audio, but it degrades faster than full large-v3 on accented and noisy speech, and is weaker on translation. On a 3070 large-v3 fp16 is already fast enough that turbo is a triage tool, not the default. Distil-Whisper (`distil-large-v3.5`) is English-only — good for a second-opinion pass, not for a multilingual pipeline.

**wav2vec2 forced alignment over Whisper's own word timestamps.** Whisper's word timings come from DTW over decoder cross-attention — usable, but drifts on long segments and around disfluencies. A CTC phoneme/character model aligned against the actual waveform gives tight, frame-accurate word boundaries, which is what makes cue timing feel professional. Fall back to Whisper DTW timestamps for languages with no alignment model.

---

## 3. Architecture

```
subtitle-generator/
├── sgen/
│   ├── cli.py                 # sgen run | reformat | qc | models
│   ├── config.py              # pydantic-settings + YAML profiles
│   ├── manifest.py            # SQLite job store: resumable, content-hash keyed
│   ├── models/registry.py     # local-path resolution, offline enforcement
│   └── stages/
│       ├── probe.py           # ffprobe: streams, duration, blake3 hash
│       ├── extract.py         # ffmpeg -> 16k mono wav (+ enhanced variant)
│       ├── enhance.py         # demucs / DeepFilterNet3
│       ├── vad.py             # Silero: speech regions
│       ├── lid.py             # language ID + optional language segmentation
│       ├── asr.py             # faster-whisper batched decode
│       ├── align.py           # wav2vec2 forced alignment
│       ├── diarize.py         # pyannote (optional)
│       ├── qc.py              # confidence, repetition/hallucination detection
│       ├── cues.py            # segmentation, line breaking, CPS, shot snapping
│       └── write.py           # pysubs2 -> srt/vtt/ass + json sidecar
├── models/                    # HF_HOME; populated once by `sgen models pull`
├── profiles/                  # accuracy.yaml, balanced.yaml, fast.yaml
└── work/                      # scratch wavs, sidecars, QC reports
```

### Data flow

```
video ──probe──> audio track choice
   └──extract──> audio.16k.wav ─────────────────┐
                      │                          │
                 (if noisy) enhance ──> audio.enh.wav
                      │                          │
                      ├──vad──> speech regions ──┤
                      ├──lid──> language(s) ─────┤
                      │                          v
                      │                    asr (batched)
                      │                          │
                      │                    segments + words
                      └──────────────> align (wav2vec2)
                                               │
                                        qc ──> flags
                                               │
                                        cues ──> write ──> .srt/.vtt/.json
```

### Process model for batch

One **GPU worker process** that loads the ASR model exactly once and consumes a queue; a **CPU thread/process pool** that runs ffmpeg extraction and enhancement for upcoming files so the GPU never waits on I/O. Prefetch depth 2 is enough. Do *not* run two ASR processes on an 8 GB card — you'll fragment VRAM and lose more to thrashing than you gain.

Every stage writes its output to `work/<hash>/<stage>.json` and records completion in SQLite. A re-run skips completed stages, so a crash at file 340 of 500 costs you one file. `sgen reformat` re-runs only `cues` + `write` from the sidecar — changing line-length rules must never mean re-transcribing.

---

## 4. Stage details

### 4.1 Audio extraction

Track selection matters more than people expect. Probe first:

```bash
ffprobe -v error -select_streams a -show_entries \
  stream=index,codec_name,channels,channel_layout:stream_tags=language:stream_disposition=default \
  -of json input.mkv
```

Choose by: `disposition:default` → matching `tags:language` → most channels → first. Skip streams whose title matches `commentary|описание|audio description`.

The default for home video downmixes to mono and levels with **`speechnorm`**:

```bash
ffmpeg -i in.mp4 -vn -sn -dn -map 0:1 \
  -af "speechnorm=e=12.5:r=0.0001:l=1,aresample=16000:resampler=soxr:precision=28" \
  -c:a pcm_s16le -ar 16000 -ac 1 out.16k.wav
```

**`speechnorm`, not `loudnorm`, is the important choice here.** `loudnorm`
targets an *integrated* loudness over the whole programme — the correct tool for
broadcast delivery, and the wrong one for this material. Given a shout at
−6 dBFS a second before a whisper at −45 dBFS, it optimises for the average and
leaves the whisper as far below the noise floor as it started. `speechnorm`
applies per-frame gain that tracks the envelope, so quiet passages come up to
where the acoustic model can actually work with them. On audio whose defining
property is uncontrolled level variation this is worth more than any decoding
parameter.

`loudnorm` remains available via `audio.normalize` for material that was
properly recorded.

Notes:
- **Normalize level, don't EQ.** Aggressive `highpass`/`lowpass` filtering
  usually *hurts* — Whisper was trained on messy web audio and expects it.
- `soxr` resampling, not the default. Cheap, and avoids aliasing on 48 → 16 kHz.
- Keep the WAV. Post-editors need the waveform, and re-alignment needs the samples.
- **Centre-channel extraction (`downmix: center`) does not apply to this
  material** and is off by default. It remains the largest single win on
  cinematic 5.1 sources — where dialogue is on a discrete channel and the L/R
  pair carries the music-and-effects bed — so it stays configurable for any
  ripped content that passes through.

### 4.2 Conditional enhancement

Enhancement is a **fallback, not a default** — it can strip vocal detail and raise WER on already-decent audio. Gate it on a measured SNR / QC failure, and always A/B:

- **Music or score under dialogue** → Demucs `htdemucs`, take the `vocals` stem. Large win on drama, trailers, vlogs with backing tracks. Costs ~1.5–2 GB VRAM and roughly 0.1–0.2× realtime on a 3070; run it *before* loading the ASR model, not alongside.
- **Stationary noise** (HVAC, hiss, traffic, fan) → **DeepFilterNet3**. Fast, low footprint, rarely harmful.
- **Reverb / phone-quality** → Resemble-Enhance can help but often over-smooths; treat as experimental.

Run ASR on both variants for flagged files and keep the one with better mean word confidence and lower repetition score. That's the automated version of what a human would do by ear.

### 4.3 VAD

Silero VAD v5 via faster-whisper's `vad_filter=True`:

```python
vad_parameters=dict(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,   # 400 for fast dialogue, 700 for lectures
    speech_pad_ms=200,             # don't clip word onsets
)
```

VAD serves three purposes: it kills Whisper's notorious hallucinations over silence and music, it defines the chunks for batched inference, and it gives `cues.py` the silence map it needs to extend short cues.

### 4.4 Language identification

Whisper's built-in detection reads only the first 30 s — which is often a logo sting, music, or a single "hello". Instead:

1. Take VAD speech regions, drop the first and last 10 % of the file.
2. Sample 8–12 chunks of 30 s spread evenly across the remainder.
3. Run detection on each. faster-whisper supports this directly:
   ```python
   model.transcribe(audio, language=None,
                    language_detection_segments=8,
                    language_detection_threshold=0.5)
   ```
4. Aggregate votes. If the top language holds ≥ 80 % of chunks → monolingual, pin `language=` for the real pass (never leave it `None` — a mid-file re-detection can flip languages and wreck a transcript).
5. If below 80 % → flag multilingual.

**Code-switching / genuinely multilingual content.** Whisper does not handle it within a segment. Do window-level LID with **SpeechBrain `lang-id-voxlingua107-ecapa`** or **`facebook/mms-lid-256`** over 3–5 s windows, smooth with a median filter or small HMM to avoid flapping, merge into contiguous language regions ≥ 5 s, then transcribe each region with its language pinned and merge results on the timeline. Emit either one mixed file or per-language tracks, configurable.

### 4.5 ASR

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel(
    r"models/ct2/large-v3-fp16",   # local path, offline
    device="cuda",
    compute_type="float16",
    num_workers=1,
)
pipe = BatchedInferencePipeline(model=model)

segments, info = pipe.transcribe(
    "out.16k.wav",
    batch_size=12,                      # 8–16 on 8 GB; drop to 6 if you co-resident Demucs
    language="de",                      # pinned from LID
    task="transcribe",
    beam_size=5,
    best_of=5,
    patience=1.0,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],   # fallback ladder
    compression_ratio_threshold=2.4,               # catches degenerate loops
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=False,              # ← important
    repetition_penalty=1.05,
    no_repeat_ngram_size=0,
    word_timestamps=True,
    hotwords="Flexoptix, QSFP28, Nagoya",          # domain vocabulary
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
)
```

Two settings deserve emphasis:

- **`condition_on_previous_text=False`.** Whisper's default of feeding prior text as context is the primary cause of runaway hallucination loops and of one bad segment poisoning everything after it. You lose a little discourse coherence and long-range consistency of proper nouns; you gain robustness. On difficult audio this is not a close call. (Batched inference disables conditioning anyway — another reason to prefer it.)
- **The temperature ladder + `compression_ratio_threshold`.** This is Whisper's built-in escape hatch: a segment that decodes to suspiciously repetitive text is re-decoded at higher temperature. Keep it enabled; it's nearly free.

`hotwords` is the cheapest accuracy win available for domain vocabulary — product names, place names, jargon, speaker names. Use it instead of `initial_prompt` (the two conflict; `hotwords` is applied more surgically and doesn't risk the model continuing your prompt's style).

**Two-tier decoding for large batches.** Pass 1: `large-v3-turbo`, `batch_size=16`, greedy. Score each segment on mean word logprob, compression ratio, and no-speech probability. Pass 2: re-decode only the bottom ~10–15 % of segments with `large-v3`, `beam_size=8`, unbatched, using neighbouring confirmed text as `prefix` for continuity. Typical outcome: ~70 % of full-large-v3 wall time at ~95 % of its accuracy. Worth building once you have a corpus large enough to care.

### 4.6 Non-speech and hallucination gating

**This is the most important stage for this material, and it is the one missing
from every off-the-shelf Whisper wrapper.**

Whisper was trained on audio paired with subtitle files. Those files contain
credits, channel boilerplate and sponsor messages that appear during passages
with no speech. The model learned the association. Given breathing, laughter,
room tone or music, it does not emit silence — it emits the text that
statistically accompanies non-speech audio:

> `Thanks for watching!` · `Untertitelung des ZDF, 2020` ·
> `Subtítulos realizados por la comunidad de Amara.org` · `[Music]`

or it locks into a repetition loop (`oh my god oh my god oh my god…`), or it
stretches two invented words across nine seconds of breathing.

None of this looks like an error in the output file. It reads as fluent,
correctly-timed dialogue that was never spoken. For personal footage that is the
worst possible failure: it fabricates a record of what people said. Suppressing
it is not a polish step, it is the correctness requirement.

[sgen/gating.py](sgen/gating.py) scores every segment and suppresses on:

| Signal | Catches |
|---|---|
| Phrase blacklist (normalized, accent-stripped, en/de/es) | Learned subtitle boilerplate |
| `compression_ratio > 2.4` | Degenerate repetition |
| Token / bigram dominance ≥ 60% | Repetition loops the ratio misses |
| All-tokens-non-lexical **and** low confidence | Breathing, sighing, throat noise |
| `no_speech_prob > 0.85` | Model itself says there is no speech |
| `avg_logprob < −1.6` | Decoded from nothing |
| `no_speech_prob > 0.6` **and** `avg_logprob < −1.0` | Weaker version of both together |
| `mean_word_prob < 0.35` | Word-level uncertainty |
| `< 0.4 words/sec` over `> 2.5 s` | Text smeared across non-speech audio |
| Text identical to 2 preceding segments | Loop across segment boundaries |

Two design decisions matter more than the thresholds:

**Non-lexical text is only suppressed when the model was *also* unsure.** A
clearly articulated "yeah" is real speech and must survive; a low-confidence
"Ah. Ah. Mmm." over breathing must not. Gating on the interjection alone would
delete real dialogue, which is why the check is a conjunction. This is the
distinction that makes the gate usable on audio full of legitimate short
utterances.

**Nothing is deleted.** Segments are marked `suppressed` with a machine-readable
reason and stay in the sidecar. `--keep-suppressed` puts them back in the
subtitle output. A gate you cannot audit is a gate you should not trust, and
these thresholds *will* need tuning against real files — the per-reason counts
printed after each run are how you do that.

The thresholds in [profiles/home-video.yaml](profiles/home-video.yaml) are
deliberately stricter than the library defaults, on the principle that for this
material a missing subtitle costs less than a fabricated one. If you find real
speech being dropped, `min_mean_word_prob` and `hard_avg_logprob` are the first
two to relax.

### 4.7 Forced alignment

```python
import whisperx, gc, torch

# free the ASR model first — 8 GB is not much
del pipe, model; gc.collect(); torch.cuda.empty_cache()

align_model, meta = whisperx.load_align_model(language_code="de", device="cuda")
aligned = whisperx.align(segments, align_model, meta, "out.16k.wav",
                         device="cuda", return_char_alignments=False)
```

- English defaults to torchaudio's `WAV2VEC2_ASR_BASE_960H` (~360 MB). Other languages use XLSR-53 fine-tunes (~1.2 GB fp32). Pre-download the ones you need.
- If the language has no alignment model, keep Whisper's DTW word timestamps and mark `alignment: "dtw"` in the sidecar so QC can weight it accordingly.
- Alignment adds roughly 0.02–0.05× realtime. Cheap.
- Sanity-check the output: alignment can fail catastrophically when the transcript and audio disagree (hallucinated text has nothing to align to). Reject a segment's alignment if word durations collapse to the frame minimum or the segment's words span < 20 % of its audio, and fall back to DTW timings for it.

### 4.8 QC

Compute per-segment and emit a report sorted worst-first — this is what makes human post-editing efficient, because the editor reviews 8 % of the file instead of scrubbing all of it:

- mean/min word probability
- compression ratio, max n-gram repetition
- `no_speech_prob`
- words-per-second outliers (> 6 wps ≈ decode error; < 0.7 wps ≈ alignment failure)
- CPS after cue building
- text that exactly repeats an adjacent segment
- known hallucination phrases ("Thanks for watching", "Subtitles by …", "♪♪") appearing over VAD-silence

Write `work/<hash>/qc.html` with clickable timestamps.

### 4.9 Cue building — the part most pipelines get wrong

Whisper's segments are not subtitle cues. They're decode windows: too long, split at attention boundaries rather than clause boundaries, and indifferent to reading speed. Build cues from **words**, not segments.

Constraints (Netflix-style Latin-script defaults, all profile-configurable):

| Rule | Value |
|---|---|
| Max lines | 2 |
| Max chars/line | 42 |
| Target reading speed | 17 CPS (20 hard max; 13–17 for children's content) |
| Min duration | 0.833 s (5/6 s) |
| Max duration | 7 s |
| Min gap between cues | 2 frames (~83 ms @ 24 fps) |
| Lead-in / lead-out | +50 ms before first word, +150 ms after last |

Algorithm:

1. **Split into sentences** on terminal punctuation from the ASR text, carrying word timings along.
2. **Greedy-pack** words into cues, breaking when the next word would exceed 2×42 chars or 7 s, or when a silence gap > 700 ms occurs.
3. **Line-break within a cue**, in preference order:
   - after terminal or clause punctuation (`. ! ? , ; :`) or an em dash
   - before a coordinating conjunction (*and, but, or, that, which*) or a preposition
   - never between an article/determiner and its noun, never inside a proper name, number, or hyphenated compound
   - otherwise the split closest to the midpoint
   - reject any break leaving a line under ~40 % of the limit (orphans read badly)
   - Where a dependency parser is available (spaCy), prefer breaks that don't split a noun or prepositional phrase. Heuristics get you ~90 % of the way; the parser handles the rest.
4. **Enforce reading speed.** If CPS > target, first extend the cue's end into following silence, bounded by `next_cue.start − min_gap`. If still over, split the cue at the best internal break point and redistribute time proportionally to character count.
5. **Enforce min/max duration**, then re-check gaps; merge cues separated by < 2 frames if their combined length fits.
6. **Snap to shot changes** (optional, high value on film/TV). Detect cuts with PySceneDetect or `ffmpeg -filter:v "select='gt(scene,0.3)',showinfo"`. If a cue boundary lands within ~12 frames of a cut, snap it to the cut; never let a cue straddle a cut by only a few frames. This is a genuine broadcast-quality differentiator and costs one cheap video pass.

Write with **pysubs2** — it handles SRT/VTT/ASS from one internal representation, so `.ass` with per-word karaoke timing for review is nearly free.

Always emit the `.sgen.json` sidecar: words with timings and probabilities, language regions, speakers, model versions, and every config value used. Everything downstream — reformatting, translation, re-timing after a re-edit — reads from it. This is what makes the pipeline maintainable.

### 4.10 Translation: document context is a capability, not a setting

The largest single quality jump in translation came from changing the **request
shape**, not the model. Both DeepL's and Google's APIs translate every item in a
request independently, so sending one cue per item means each line is translated
as though it were the only sentence in existence:

| source | one cue per item | whole numbered transcript per request |
|---|---|---|
| `мастер по интернету` | web developer | The internet technician |
| `отрабатывает` | working | He's working on it. |
| `Машка, машка, машка.` | Mouse, mouse, mouse. | Masha, Masha, Masha. |
| `Музыка Секунду.` | Music by Sekunda. | Music. Just a second. |

So the cloud path sends the transcript as numbered lines in one request and maps
the numbers back onto the timings — the manual copy-paste workflow, automated.
It verifies that ≥80% of cues returned with their number and falls back to
per-cue otherwise: **alignment beats fluency**, because a fluent subtitle on the
wrong timing is worse than a clumsy one on the right timing.

**Do not try this with the local model.** NLLB-200 is trained on single sentence
pairs and has no document-level capability; the same input shape that improves a
cloud service destroys it. Measured on an 84-cue transcript:

- whole document → output stopped at line 11, losing 73 cues. Raising
  `max_decoding_length` from 512 to 2048 changed nothing: the model emits EOS
  there by itself.
- 2, 3 and 5 numbered lines → **0 cues** mapped back, because it translates the
  numerals into words ("1." → "One,").
- `Тихо, может звонит. Да, любимый? Да, все хорошо.` → *"I don't know, it's a
  phone call."* Three sentences in, one wrong sentence out.

Hence `split_sentences()` and the 200-char / 8-second cap in `group_sentences()`:
they are not conservatism, they are the largest input this model survives.
Improving local translation therefore means replacing the model with a
document-capable one, not reshaping its requests.

### 4.11 Two scripts, two completely different problems

Writing a transcript in a second alphabet looks like one feature with a
parameter. It is two, and they fail in opposite directions.

**Devanagari → Latin is many-to-one.** Every Devanagari letter has one sensible
Roman spelling. The work is all in readability conventions — deleting the
word-final inherent vowel (`raam`, not `raama`), reading फ as /f/, six
conventional spellings — and a mistake produces something slightly odd but
still legible. `jitani` for जितनी is imperfect and harmless.

**Devanagari → Urdu is one-to-many.** Urdu keeps Perso-Arabic orthography for
Perso-Arabic vocabulary: /z/ is ز, ذ, ض or ظ; /s/ is س, ص or ث; /t/ is ت or ط —
selected by the word's etymology, not by its sound. Devanagari discarded those
distinctions centuries ago, writing स for all three /s/ letters and क for both
/k/ and /q/. **The information needed to spell correctly is not in the input.**

So no table can be right, and the design admits it rather than hiding it:

- a **word list** for Perso-Arabic vocabulary, which in film dialogue is most of
  the content words (`حق`, not the phonetically-correct-and-wrong `ہک`)
- normalisation before lookup, because Whisper rarely writes nuktas and varies
  between anusvara and a nasal consonant: मंज़िल, मंजिल and मन्जिल must all find
  the same entry
- a **future-tense rule**, because Urdu writes it as two words with a nasal
  (जाऊँगा → `جاؤں گا`) where Devanagari joins them, and nearly every line of a
  song lyric is a future verb. No letter table can produce that space.

A word outside the list comes out phonetically right and orthographically naive.
That is stated in the tooltip and the README rather than left for the user to
discover, because the failure is invisible to anyone who cannot already read the
script — which is precisely the person the feature is for.

The same experiment on Russian and German is instructive: those have **no correct
spelling to miss**, since a foreign word in Urdu script is a phonetic respelling
by definition. The linguistically harder pair is the easier program. They are not
shipped because "readable" is doing a lot of work there and nobody actually reads
Russian in Urdu letters.

### 4.12 The cache is a record, so forgetting is a feature

`work/<content-id>/` exists for performance: it is what makes `reformat`,
retiming and translation free, and what lets a re-added file skip the GPU
entirely. But look at what is in it. `transcript.sgen.json` contains the full text
of everything said in the room and the absolute path it was said in;
`audio.16k.wav` contains the audio. For the material this pipeline is aimed at —
home recordings of identifiable people — that is a more sensitive artifact than
the subtitles it was built from.

Two consequences shape the design.

**Deleting an entry is a first-class action, not a documented folder path.** It is
a button on every row in the UI and a `sgen forget` command, because a privacy
control that requires knowing the layout of the project directory is a privacy
control most people will never use.

**Forgetting stops at the cache.** It deletes `work/<id>/` and nothing else: the
`.srt` next to the video was explicitly asked for, and deleting the thing someone
came here to produce in the name of privacy would be a worse surprise than
leaving the cache. The split is "what the app kept for itself" versus "what the
user asked for".

`--all` additionally clears folders from runs that never finished. Those hold
extracted audio with no sidecar, so they appear in no list — which makes them
exactly the thing a privacy sweep must not miss.

---

## 5. VRAM budget on 8 GB

| Configuration | Weights | Peak w/ activations | Fits? |
|---|---|---|---|
| large-v3 fp16, batch 8 | ~3.1 GB | ~4.5–5 GB | ✅ comfortable |
| large-v3 fp16, batch 16 | ~3.1 GB | ~6–6.5 GB | ✅ close the browser |
| large-v3 int8_float16, batch 16 | ~1.6 GB | ~4 GB | ✅ lots of headroom |
| large-v3-turbo fp16, batch 16 | ~1.6 GB | ~3.5 GB | ✅ |
| + wav2vec2 XLSR align, concurrent | +1.2 GB | — | ⚠️ free ASR model first |
| + Demucs htdemucs, concurrent | +2 GB | — | ❌ run as a separate stage |
| + pyannote 3.1, concurrent | +1.5 GB | — | ⚠️ sequence it |

**Rule: one model resident at a time.** Explicit `del model; gc.collect(); torch.cuda.empty_cache()` between stages. Windows WDDM also lets the desktop compositor take 600–900 MB, which is already visible on this machine — budget for ~7 GB usable, not 8.

Rough throughput on a 3070 (varies with audio density; measure yours):

| Config | Speed |
|---|---|
| large-v3 fp16, sequential, beam 5 | ~6–10× realtime |
| large-v3 fp16, batched (12) | ~20–35× realtime |
| large-v3-turbo fp16, batched (16) | ~60–90× realtime |
| + forced alignment | −5 to −10 % |

So a 45-minute episode at the accuracy default lands around 1.5–2.5 minutes.

---

## 6. Setup on this machine

```powershell
# ffmpeg (currently missing)
winget install Gyan.FFmpeg

python -m venv .venv; .\.venv\Scripts\Activate.ps1

# PyTorch CUDA 12.x wheels — fine on a CUDA 13.2 driver (drivers are backward compatible)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install faster-whisper pysubs2 pydantic-settings typer rich blake3
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # see note below
pip install whisperx                                       # for align only
pip install deepfilternet demucs                           # optional enhancement
pip install speechbrain                                    # optional multilingual LID
```

**The cuDNN trap.** CTranslate2 ≥ 4.5 links cuDNN 9; earlier versions want cuDNN 8. Symptom: `Could not locate cudnn_ops_infer64_8.dll` or `cudnn64_9.dll`. Fix by installing the pip cuDNN package above and adding its DLL directory to the process path before importing:

```python
import os, nvidia.cudnn.lib
os.add_dll_directory(os.path.dirname(nvidia.cudnn.lib.__file__))
```

Pin `ctranslate2` and `faster-whisper` together in your lockfile — mismatches here are the most common install failure on Windows.

**Model acquisition (once, then offline):**

```powershell
$env:HF_HOME = "C:\repos\subtitle-generator\models"

# Pre-converted CT2 weights
huggingface-cli download Systran/faster-whisper-large-v3
huggingface-cli download deepdml/faster-whisper-large-v3-turbo-ct2

# Or convert any HF Whisper yourself — avoids depending on third-party repos
pip install transformers[torch]
ct2-transformers-converter --model openai/whisper-large-v3 `
  --output_dir models\ct2\large-v3-fp16 --quantization float16 `
  --copy_files tokenizer.json preprocessor_config.json

# Alignment models for the languages you need
python -c "import whisperx; [whisperx.load_align_model(l,'cpu') for l in ['en','de','fr','es']]"
```

Then set `HF_HUB_OFFLINE=1` permanently. Add a `sgen models verify` command that fails loudly if anything would need the network — that's how you actually guarantee the offline property rather than assuming it.

`pyannote/speaker-diarization-3.1` requires accepting terms on Hugging Face and a token for the *initial* download; it runs offline afterwards. If that's unacceptable, skip diarization — it's orthogonal to subtitle quality unless you need speaker labels.

---

## 7. Batch processing

```powershell
sgen run --profile accuracy --glob "D:\media\**\*.mkv" --out-dir "D:\subs" --jobs-cpu 4
```

- SQLite manifest keyed by content hash → idempotent, resumable, safe to re-run over a growing library.
- Stage-level caching: changing the cue profile re-runs only `cues` + `write`.
- CPU pool prefetches extraction 2 files ahead of the single GPU worker.
- Per-file failures are recorded and skipped, never fatal to the batch. Print a summary table at the end with QC scores so you can see which 20 files need attention.
- Write outputs next to the video (`movie.en.srt`) or into a mirrored tree — configurable, since players expect the former and archives want the latter.

## 8. Post-editing workflow

**Subtitle Edit** (Windows, GPLv3) is the right tool and the reason to keep the WAV around:

1. Ship `movie.mkv`, `movie.en.srt`, `movie.16k.wav`, `movie.sgen.json`, `qc.html` together.
2. Open the SRT in Subtitle Edit; it renders the waveform, shows a live CPS column that goes red on violations, imports shot-change lists, and has a solid "Fix common errors" pass.
3. Work the QC report worst-first rather than linearly.
4. Corrections go back into the sidecar via `sgen import-srt`, so a later re-break or re-format doesn't discard human edits.

Aegisub is the alternative if you're doing ASS styling or typesetting.

---

## 9. Practical tips for difficult audio

Ordered by expected payoff **on home-video material**:

1. **Gate the output (§4.6).** On audio containing breathing, whispering and
   non-lexical vocalization this is worth more than every decoding parameter
   combined, because it converts confident fabrication into an honest gap.
2. **`speechnorm`, not `loudnorm`.** Per-frame levelling is what makes a
   whispered passage audible to the model at all. Integrated loudness
   normalization averages it away. See §4.1.
3. **`condition_on_previous_text=False`.** Stops one hallucinated patch from
   poisoning the rest of the file — and on this material there *will* be
   hallucinated patches.
4. **`hotwords`** with the names of the people, places and pets that appear in
   your footage. Minutes of effort, and it fixes exactly the errors that matter
   most in personal recordings: names of people you know.
5. **Keep the temperature ladder and compression-ratio threshold on.** Free
   hallucination insurance, already wired up.
6. **VAD with `speech_pad_ms=200`.** Whispered onsets are quiet and unvoiced;
   a tight pad costs you the first phoneme of every cue.
7. **Window-level LID (§4.4) for mixed-language files.** Pin one language per
   region; never let Whisper re-detect freely mid-file, which produces
   transcripts that flip language mid-sentence.
8. **Expect to tune the gate thresholds** against your own files, using the
   per-reason counts printed after each run. Ship-and-tune beats guessing: the
   defaults are a starting point calibrated on assumptions, not on your footage.
9. **Try `beam_size=8` with `patience=2`** on your worst files (the `forensic`
   profile). Wider beam search genuinely helps when the acoustic evidence is
   ambiguous — which is the definition of this material.
10. **Second-opinion cross-check** where it matters: decode again with a
    different model and flag disagreements. Disagreement is a strong error
    signal even when you can't tell which output is right — ideal for routing
    review attention.
11. **Don't over-denoise.** Gate any enhancement on measured SNR and validate by
    comparing confidence on both variants. Denoisers strip speech detail along
    with noise, and this is doubly true for whispered speech, where the signal
    *is* breath noise — a denoiser can remove the very thing you need.
12. **Demucs only if there's actually music.** It's a large win on a music bed
    and a waste of 2 GB of VRAM otherwise.
13. **Accept that some passages have no correct answer.** Where humans can't
    make out the words, the right output is a flagged gap, not a guess. Budget
    review time rather than more GPU passes.
14. **Long files:** process in VAD-bounded blocks of ~10–15 minutes with a few
    seconds of overlap, stitching on word boundaries. Avoids memory growth and
    localizes any failure.
15. **Verify with the alignment**, not the transcript. Words whose aligned
    duration is implausible for their length are the reliable tell for
    hallucinated text that reads fluently.

---

## 10. Profiles

| Profile | ASR | Batch | Beam | Align | Enhance | Speed |
|---|---|---|---|---|---|---|
| `fast` | turbo fp16 | 16 | 1 (greedy) | DTW only | off | ~60–90× RT |
| `balanced` | turbo → large-v3 rescue | 16 / 1 | 1 / 8 | wav2vec2 | on if flagged | ~30–50× RT |
| `accuracy` | large-v3 fp16 | 8 | 8 | wav2vec2 | on if flagged | ~15–25× RT |
| `forensic` | large-v3 fp16 + second model | 1 | 8, `patience=2` | wav2vec2 | forced | ~4–6× RT |

Default to `accuracy` for anything a human will read, `fast` for search indexing and rough cuts.
