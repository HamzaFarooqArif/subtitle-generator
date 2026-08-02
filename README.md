# sgen — offline subtitle generator

Local, GPU-accelerated subtitle generation for video files. No cloud APIs; after
a one-time model download the pipeline runs with the network hard-disabled.

Built for an RTX 3070 (8 GB) and tuned for **difficult consumer audio** rather
than broadcast material: built-in microphones, a wide dynamic range inside one
file, background noise, low-energy and ambiguous speech, non-speech
vocalization, and mid-file language switching.

See [DESIGN.md](DESIGN.md) for the reasoning behind the model, quantization and
architecture choices.

## Install

```powershell
winget install Gyan.FFmpeg

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -m sgen doctor          # check ffmpeg, CUDA, cuDNN, models
python -m sgen models pull     # one-time download (~7 GB) — the only online step
python -m sgen models verify   # prove everything resolves with the net disabled
```

## Use — web UI

```powershell
python -m sgen ui
```

Opens `http://127.0.0.1:8420`. **Localhost only, and media is never uploaded** —
you pick paths in a server-side file browser and the pipeline reads them where
they already are, so nothing is copied and gigabyte files never touch the wire.

Three tabs:

Every control carries a **?** that says what it does in the terms that decide it:
for a checkbox, what is different when it is on; for a slider, what happens at
each end; with a worked example where one helps. Hover, focus it with the
keyboard, or click to pin it. A control with no help text fails the test suite.

**Subtitles** — browse to your files, set the profile / language / hotwords, and
submit. Jobs run one at a time on a single GPU worker (the model loads once and
stays loaded between jobs), with live per-stage progress streamed over SSE.
Finished files, and everything transcribed in earlier sessions, are listed
underneath with a **Translate…** button.

**Folder mode** sits under the file browser: **Check this folder** reports what
it still needs, **Transcribe folder** queues only that. Interrupt it however you
like — close the app, restart the machine — and run it again; it continues.

Clicking files and ticking a scan are the two ways to choose what runs, and only
one can be active: starting either lets go of the other, and says so. Both at
once meant the ticks were silently ignored in favour of the clicked files.

Whether a video is finished is decided from **the subtitle files next to it**,
never from anything the app remembers, because a job list in memory cannot
survive a reboot and a database can disagree with the disk. Three things make
that safe:

- subtitles are written to a temporary file and renamed into place, so a file
  that exists is a file that finished
- the language is in the filename, so a translation that was requested and never
  produced shows up as an absence — `clip.de.srt` without `clip.en.srt`
- a subtitle file with no readable final cue is treated as **interrupted** and
  produced again rather than trusted

```
4 media file(s) · 2 already done · 1 needing translation only · 1 interrupted, will be redone
  one.mp4      done      en subtitles present
  four.wav     done      de, en subtitles present
  three.mp4    damaged   1 subtitle file(s) look truncated — probably interrupted
  two.mp4      translate transcribed as de but no en translation
```

The CLI resumes identically: `python -m sgen run "D:\videos"` skips finished
files and says so; `--no-resume` redoes everything.

**Settings for one file.** A folder is rarely uniform: two files are songs, one
is already in English, one wants Latin-script output. Each row in the scan has a
**Settings…** button that opens that file on a second tab of the settings panel —
the same controls, each offering *as in All files (…)* so an override is visibly
an exception rather than a separate set of settings. Anything left inherited
follows the first tab.

**Nothing is written until you press Save** (or Ctrl+S), and leaving with unsaved
edits asks first — switching tabs, opening another file, changing folder, closing
the page, or starting a run, since a run would not use them. Cancel keeps you
where you are; **Discard changes** puts the form back.

They are saved in `sgen.folder.yaml` **beside the videos**, for the same reason
resumability reads the subtitle files: state in the app cannot survive a restart,
and a database can disagree with the disk. It is meant to be hand-editable — for
fifty files that beats fifty clicks:

```yaml
files:
  "Full Song - KHAIRIYAT.mp4":
    profile: music
    romanize: true
  "beach 2019.mp4":
    translate: none        # already English
  "interview.mp4":
    language: auto         # detect this one, even though All files pins German
```

`Reset N files to All files` in the scan's action row deletes the lot.

**Translate** — every transcript you have, each with a Translate button:
Google or DeepL if you have a key, or the paste-it-yourself round trip if you
don't. Works from the stored transcript, so no GPU and no second pass.

**Tune gate** — sliders for every gate threshold that **re-run gating and cue
building from the stored transcript with no GPU and no re-transcription**. Move a
threshold and the kept/suppressed counts, the per-reason breakdown, and the full
list of what got cut update immediately. This is how you calibrate the gate
against your own footage instead of trusting defaults. `Save as profile default`
writes the values back into the profile YAML.

**Starting it twice replaces the running app** rather than adding to it. Without
that, the old server keeps the port, the new one lands on another, and you end up
with several instances serving current HTML from disk against stale API handlers
— a mismatch where the page looks right and behaves wrong. Only processes that
answer as sgen are stopped: anything else on the port is reported and left alone.

```powershell
python -m sgen stop            # stop everything, start nothing
python -m sgen ui --no-replace # leave previous servers running
```

Options: `--port`, `--no-open`, `--host`, `--replace/--no-replace`, `--reload`.

## Use — CLI

```powershell
# One file, subtitles written next to it
python -m sgen run "D:\videos\clip.mp4"

# A whole tree, into a separate output directory
python -m sgen run "D:\videos" -o "D:\subs"

# Pin the language when you already know it (faster and safer than detection)
python -m sgen run "D:\videos\clip.mp4" --language de

# Bias decoding toward names the model won't guess
python -m sgen run "D:\videos" --hotwords "Thomas, Oaxaca, Kreuzberg"

# See what the gate removed instead of trusting it
python -m sgen run "D:\videos\clip.mp4" --keep-suppressed

# Re-break lines without re-transcribing (no GPU needed)
python -m sgen reformat work\<id>\transcript.sgen.json --max-chars 32
```

## What it produces

For `clip.mp4` it writes `clip.en.srt` beside the source, plus a sidecar at
`work/<content-id>/transcript.sgen.json` holding every word with its timing and
confidence, the gating decisions, and the full config used. WebVTT is available
too — tick `.vtt` in the UI, or add it to `defaults.formats` — but off by
default, since beside a video file it is a second copy of the same subtitles.

**The sidecar is the source of truth.** Reformatting, retiming and future
translation all read from it, so changing subtitle style never means paying for
another transcription pass.

### Forgetting a file

That cache is also a record: the sidecar holds the full text of what was said and
the path it came from, and `audio.16k.wav` holds the audio itself. For personal
footage that matters, so it is removable without knowing which folder to open.

On the **Translate** tab each file has a **Forget** button (two clicks, since the
transcript cost GPU time and the source may be gone), plus **Forget everything**
for the whole cache. From the terminal:

```powershell
python -m sgen forget                 # what is cached, how big, and its ids
python -m sgen forget 06f677d0590710db
python -m sgen forget --all
```

Both delete `work/<id>/` — transcript, audio and any hand edits. **Your subtitle
files, next to the video, are left alone**: they are what you came here for.
`--all` also clears folders left by runs that never finished, which hold extracted
audio but appear in no list.

## Profiles

| Profile | For | Key difference |
|---|---|---|
| `home-video` (default) | Handheld mics, uneven levels, unclear speech | `speechnorm` levelling, strict gating |
| `music` | Songs, or speech under dense instrumentation | **VAD disabled**, looser gating, slower reading speed |
| `verbatim` | **"Subtitles are missing things I wanted"** | VAD off, confidence gating off — keeps everything, you delete by hand |

**If text is missing, try `verbatim` first.** Measured on a 27-minute file whose
vocalization is largely non-lexical: `home-video` kept 84 cues and suppressed 51%
of segments, `verbatim` kept 111 cues and suppressed 4%. Much of what
`home-video` removed was correct (breathing, repeated "ah"), but it also removed
genuine short utterances at low confidence and genuine repetition that the
repeat-loop check cannot tell apart from a decode loop.

**Use `music` for anything sung.** Silero VAD is trained on speech and does not
recognise singing over a full arrangement: on a 239-second Bollywood track it
classified **0.7 seconds** as speech. The pipeline now detects this and retries
automatically, but starting from the `music` profile skips the wasted first pass.

## Settings

`settings.local.yaml` in the repo root holds everything specific to your
machine: API keys, which profile to start on, where subtitles go, the UI port.
It is gitignored, so keys are safe in it. `settings.example.yaml` is the
commented template.

```powershell
sgen config --init                          # create it from the template
sgen config                                 # show what is in effect, and from where
sgen config --set defaults.profile=music    # or just edit the file
sgen config --edit                          # open it in your editor
```

```yaml
api_keys:
  google: "AIza…"          # for the Translate button; nothing else reads it
  deepl: ""
  deepl_plan: free         # free keys end in ":fx" and are detected anyway

defaults:
  profile: home-video
  language: ""             # "" detects per file — leave it for mixed-language audio
  hotwords: "Thomas, Oaxaca, Kreuzberg"
  romanize: false
  keep_suppressed: false
  formats: [srt]           # add vtt only if you need it for a browser player
  out_dir: ""              # "" writes next to each source file
  translate:
    auto: false            # true: translate anything not already in `target`
    provider: google       # google | deepl | local (the offline model)
    target: en

server:
  host: 127.0.0.1          # localhost only — see the note in the template
  port: 8420
  open_browser: true
  replace_running: true    # stop a previous server instead of stacking another
```

Everything is optional; delete a line and the built-in default applies. A typo
is reported rather than ignored — a silently dropped setting is
indistinguishable from a setting that doesn't work — and a broken file never
stops a transcription: the pipeline needs none of this.

Precedence, highest first: **a flag or UI control** you set for this run →
**environment** (`SGEN_GOOGLE_API_KEY`, `SGEN_DEEPL_API_KEY`, for keys you would
rather not write to disk) → **`settings.local.yaml`** → **the profile** →
built-in defaults. Profiles win over settings for anything the settings file
doesn't mention, so setting `formats` here overrides a profile while leaving
alone one that customizes it.

The file is re-read on every request, so editing it while the UI is running
takes effect immediately — no restart. Saving a key from the UI rewrites only
that one line: your comments, ordering and other settings survive, because a
hand-maintained file that a program overwrites wholesale stops being worth
maintaining by hand.

## Translation: use an external translator

**Honest position: the local translation models do not match Google Translate**,
and for Hindi they are not close. Transcription quality is a different story —
that runs locally and is good — but translation is where a production service
with vastly more training data wins, and no 8 GB local model changes that.

### Automatic, via Google or DeepL (recommended)

Two ways in, both using the same code so the output is identical:

- **While transcribing** — Settings → **Also translate** → `Cloud: DeepL` or
  `Cloud: Google Translate`. The translation runs after the subtitles are on
  disk, so a rejected key costs you a translation, never the transcription; the
  outcome is reported on the file's row either way. **Set up cloud API keys…**
  next to it goes straight to the key fields.
- **Always, for anything not already in English** — tick **Always do this** and
  the choice is written to `defaults.translate.auto` in `settings.local.yaml`, so
  it applies to every later run and survives a restart. Files whose detected
  language is already the target are skipped rather than sent — most home video
  is already English, and translating it would spend quota to get the same words
  back. The row says `already in en — nothing to translate`.
- **Afterwards** — the **Translate** tab lists every transcript you have. Pick
  one, choose a provider and a language, press **Translate now**.

| | Free tier | Notes |
|---|---|---|
| **Google Cloud Translation** | 500k chars/month | Needs a Cloud project with the Translation API enabled and billing active. Best coverage — solid on Hindi and Russian. |
| **DeepL** | 500k chars/month | Often better on register and idiom. Its language list is read from the API at runtime (110 targets, including Hindi and Urdu) rather than hardcoded — a written-down list went stale and refused pairs the service supported. |

500k characters a month is a lot for this: a 25-minute video's transcript is
about **1,600 characters**, so roughly **300 files a month inside the free tier**.

Keys live in `settings.local.yaml` (see [Settings](#settings)), which is
gitignored. They are never written to a sidecar or a subtitle file, and never
appear in an error message. Nothing online is contacted during a normal
transcription — only when you press Translate now.

Where to get a key:

- **Google** — [create a project](https://console.cloud.google.com/projectcreate)
  → [enable the Cloud Translation API](https://console.cloud.google.com/apis/library/translate.googleapis.com)
  → [link billing](https://console.cloud.google.com/billing)
  → [Credentials → Create credentials → API key](https://console.cloud.google.com/apis/credentials).
  A card is required even for the free allowance. Worth restricting the key to
  the Translation API and setting a budget alert.
- **DeepL** — [sign up for DeepL API Free](https://www.deepl.com/pro-api)
  → [Account → API keys](https://www.deepl.com/your-account/keys). Free keys end
  in `:fx`, which selects the right hostname automatically.

The **Test** button next to the key field translates one word, so a bad key
fails in a second instead of halfway through a file.

### Or paste it yourself

If you'd rather not use a key, the manual round trip is still there and keeps
everything except the translation itself local:

1. Press **Translate…** on a finished file
2. **Copy subtitle text** — numbered, one line per cue
3. Paste into Google Translate (or anything else), copy the result
4. Paste it back and **Apply translation**

Your timings, line breaking and reading-speed rules are all reapplied here, so
the only thing outsourced is the wording. Lines are numbered because translators
merge, split and drop lines; numbering survives that and each translation is
matched back to its own cue. If numbers come back stripped, it falls back to
positional matching. Any line that can't be matched keeps its original text and
is flagged `untranslated` rather than silently vanishing.

From the CLI:

```powershell
python -m sgen export-text work\<id>\transcript.sgen.json --out text.txt
#  ... translate text.txt however you like ...
python -m sgen import-text work\<id>\transcript.sgen.json translated.txt -o D:\subs
```

**Privacy note:** pasting text into an online translator sends it to that
service. That's fine for a film or a song. For personal recordings it's worth a
deliberate decision — the transcription itself never leaves the machine, and this
step is the one place that changes.

### If you want a better local option

`ai4bharat/indictrans2-indic-en-1B` is purpose-built for Indic↔English and
substantially outperforms NLLB on Hindi. It's a **gated** Hugging Face repo, so
it needs your account: accept the terms on the model page, then
`huggingface-cli login`. Tell me once that's done and I'll wire it in.

## Built-in translation (weaker, fully offline)

Tick **"Also write English subtitles"**, or pass `--translate`. Written as
`name.en.srt` alongside the native-language file. `--translate-to de` targets a
different language.

There are two engines and **neither wins everywhere**, so the default (`auto`)
measures the transcript and picks:

| Engine | How | Best at | Needs |
|---|---|---|---|
| `nllb` | Translates the transcript **text**, sentence by sentence | Ordinary speech | NLLB model (`sgen models pull --translation`) |
| `whisper` | Translates the **audio** in 30-second chunks | Text with no sentence punctuation | Nothing extra |

Measured, same pipeline, same files:

**Conversational German** — NLLB clearly better:
```
nllb    : Today we're going down to the river to look at the old pier.
whisper : today we go down to the river and look at the Alte Proecke  (untranslated)
```

**Unpunctuated Hindi song lyrics** — Whisper clearly better:
```
whisper : Ask about the good, sometimes ask about the bad
nllb    : Ask him how is your heart without Divane?  (repetitive, untranslated name)
```

The reason is structural: text translation needs sentence boundaries to work
with, and sung lyrics have none, so it rambles and repeats. `auto` therefore
routes on the fraction of segments ending in sentence punctuation — under 30%
goes to Whisper. The choice is logged per file.

Because `nllb` works on text it also translates your **hand-edited** subtitles,
runs with no audio decode, and can target languages other than English.

Other things to know:

- **Timings differ between engines.** Whisper decodes the audio again, so its
  English cues don't line up frame-for-frame with the native ones. NLLB keeps
  the original timings.
- Translation is never gated on the source audio's confidence. Those numbers
  describe the speech, not the translation; judging the translation by them
  suppressed 40% of a good result and truncated lines mid-sentence.
- NLLB translates one sentence at a time and **silently drops everything after
  the first sentence boundary**, so inputs are split into sentences first.
  Without that, "Where are you going? Come with me." came back as "Where are you
  going?"

All three outputs from one run:

```
song.hi.srt        खैरियत पूछो कभी तो कैफियत पूछो
song.hi-Latn.srt   Khairiyat puchho kabhi to kaifiyat puchho
song.en.srt        Ask about the good, sometimes ask about the bad
```

## Latin-script subtitles (romanization)

For languages you speak but don't read: tick **"Also write Latin-script
subtitles"**, or pass `--romanize`. नमस्ते becomes `namaste`. This is
transliteration, not translation — same words, different letters — written as a
second file so you keep both:

```
song.hi.srt        खैरियत पूछो कभी तो कैफियत पूछो
song.hi-Latn.srt   Khairiyat puchho kabhi to kaifiyat puchho
```

Supported: Hindi, Marathi, Nepali, Sanskrit, Bengali, Assamese, Punjabi,
Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Sinhala. Other scripts pass
through unchanged.

Hindi output follows what a Hindi speaker would actually type rather than a
scholarly scheme — [sgen/translit.py](sgen/translit.py) deletes the word-final
inherent vowel (`raam`, not `raama`), reads फ as `f` (`kaifiyat`), and leaves
English words in mixed Hinglish text untouched.

**Known limitation:** only *word-final* schwa deletion is applied, so medial
cases come out slightly long — जितनी is `jitani` where you would write `jitni`.
Readable, but not exactly idiomatic.

## Mixed-language files: leave language on auto-detect

Measured on a fixture alternating English and German sentences:

| Setting | Coverage | Outcome |
|---|---|---|
| Auto-detect | 84% | All sentences correct, in both languages |
| Pinned `de` | 54% | **Both English sentences dropped entirely** |
| Pinned `es` (absent from the audio) | 84% | All sentences correct |

Whisper's language token biases decoding rather than hard-constraining it, so a
detected language still lets other languages through — but an explicit pin
suppresses them. **Pin the language only when the whole file is one language and
detection is demonstrably getting it wrong.** Check the reported confidence
first; the pipeline warns below 50%.

## When a result is not credible

Per-segment gating cannot catch a file that failed wholesale — the one or two
segments that survive look locally fine. So [sgen/qc.py](sgen/qc.py) judges the
file as a whole and flags it as **suspect**, loudly, in both the CLI and the UI:

- subtitles cover an implausibly small fraction of the audio
- speech detection found almost no speech (the music case above)
- the language was detected at low confidence — a wrong guess yields fluent
  nonsense in the wrong language, which is far worse than an obvious error
- too few segments for the duration, or most segments were gated

This exists because of a real failure: a 4-minute Hindi song produced a single
0.5-second cue of Turkish text and was reported as a **success**. Never again —
a file that essentially failed now says so.

## The gating stage

On breathing, whispering and non-speech vocalization Whisper does not return
silence — it returns fluent, confident, invented text ("Thanks for watching",
repeated phrases, subtitle-credit boilerplate it absorbed from training data).
A plausible-looking fabricated subtitle is worse than a missing one, so
[sgen/gating.py](sgen/gating.py) suppresses segments that look non-lexical or
hallucinated, using confidence, repetition, word-rate and a phrase blacklist.

Nothing is deleted: suppressed segments stay in the sidecar with a reason, and
`--keep-suppressed` puts them back in the output so you can audit the gate.

Expect to tune the thresholds in [profiles/home-video.yaml](profiles/home-video.yaml)
once you have run real files through it. Start by looking at the reason counts
printed after each file.

## Tests

```powershell
pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

The gating and cue-building tests are pure logic and need no GPU. The
end-to-end test needs a fixture, which is generated from the built-in Windows
speech voices so no media is checked in:

```powershell
powershell -ExecutionPolicy Bypass -File tests\make_fixture.ps1
```

The fixture deliberately contains a clean spoken passage, a block of pink
noise, and the same speech at −22 dB — the noise and quiet passages are what
the gating and normalization stages exist to handle.

## Status

**Working, and exercised on real footage** (436 tests, 9 of them end-to-end):

- probe and audio-track selection, extraction with `speechnorm` levelling
- per-file language detection, with a no-VAD retry when speech detection rejects
  almost the whole file (sung audio does this; measured 0.7 s of 239 classified
  as speech)
- ASR on `large-v3` float16, sentence-level resegmentation, non-speech and
  hallucination gating, file-level QC verdict
- cue building with clause-aware line breaks, reading-speed enforcement and
  orphan control; `.srt`/`.vtt` output, UTF-8 with BOM
- Latin-script transliteration for Indic scripts (नमस्ते → `namaste`)
- translation: Google or DeepL through the API, sending the numbered transcript
  as one document (see [DESIGN.md §4.10](DESIGN.md)); offline NLLB-200 as the
  privacy-preserving alternative; and a manual paste round trip needing no key
- local web UI: file browser, queue with live per-stage progress over SSE, a
  library read from sidecars so nothing becomes unreachable after a restart, and
  gate-threshold tuning that rebuilds cues without touching the GPU
- forgetting a file: deleting the cached transcript and audio from the UI or the
  CLI, without touching the subtitles that were asked for
- batch processing with the model loaded once per run

**Not built:**

- wav2vec2 forced alignment — weights download, the stage is not wired in
- window-level LID, so a file that switches language mid-way is decoded as one
  language throughout. Detection per file works well; **pinning** a language on a
  mixed file is worse than leaving it automatic (measured 54% vs 84% coverage on
  an English/German file), which is why the UI advises against it
- Demucs / DeepFilterNet enhancement
- speaker diarization
- SQLite manifest for resumable batches (stage caching is content-hash
  directories only)

**Known rough edge:** progress within the *transcribe* stage is coarse. Batched
inference yields segments in bursts, so on short files the bar sits still and
then jumps. Stage transitions are accurate; the fraction inside the longest
stage is not always informative.

**Partly tuned:** the gate thresholds in
[profiles/home-video.yaml](profiles/home-video.yaml) have been checked against
real files — see the measured comparison in [Profiles](#profiles) — but not
swept systematically. The failure mode to watch for is real speech being
suppressed, which shows up as high `low_word_confidence` or `non_lexical` counts
in the per-run output. The **Tune gate** tab exists to calibrate this against
your own material without re-transcribing.

## Licence

MIT — see [LICENSE](LICENSE).
