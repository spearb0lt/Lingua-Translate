# Lingua

Translation for Indian and world languages, built to run on your own hardware
first and to fall back to free services only when it has to.

Three things make it different from a thin wrapper around a translation API:

**It reads romanized Indic text.** `mera naam rahul hai` is Hindi typed in
Latin letters. Lingua detects that, converts it to `मेरा नाम राहुल है`, shows
you what it understood, and translates from there. This is not cosmetic —
measured against Google's own endpoint, transliterating first is strictly
better:

| input | sent as typed | sent as native script |
|---|---|---|
| `ami tomake bhalobashi` (bn→en) | *returned untranslated* | **i love you** |
| `naan unnai kaadhalikkiren` (ta→hi) | `नान उन्नै कधलिक्किरें` — transliterated, not translated | **मुझे तुमसे प्यार है** |
| `amar nam gobindo` (as→en) | `love me gobindo` | **Our name is Govinda** |

That last row is the exact example from
`translate.google.co.in/?sl=as&tl=en&text=amar%20nam%20gobindo` — Lingua beats
what Google's own UI returns for it, because it fixes the script first.

**It prefers local weights.** IndicTrans2 (MIT) for English ↔ 22 Indian
languages and Indic ↔ Indic; NLLB-200 for everything else; a two-hop English
pivot for Indic ↔ foreign, which measurably beats NLLB direct. All run on a
4 GB laptop GPU or CPU-only. Nothing leaves the machine unless a local model
can't serve the pair.

**It degrades instead of failing.** Every request walks a tier ladder and the
UI tells you which rung answered — and it checks its own output, so a garbled
translation is rejected rather than served.

**It can tell you how much to trust it.** Ask for an accuracy check and the
result is translated back through a *different* engine and scored, with the
round trip shown so you can judge it yourself. Ask for best-of-3 and it runs
three engines and keeps the highest-scoring one.

**It translates whole documents**, not just sentences — Word, PDF, subtitles
and plain text, keeping timings, styling, tables and indentation.

---

## The tier ladder

```
Tier 0  LOCAL      IndicTrans2 · pivot_en · NLLB-200   open weights, on your machine
Tier 1  FREE_WEB   Google gtx · Google clients5        keyless, no account
Tier 2  FREE_API   MyMemory · LibreTranslate           free tiers, documented
Tier 3  FREEMIUM   only if you configure a key         off by default
```

Speech has its own chains, listed under [Coverage](#coverage).

Lowest tier that can serve the pair wins. If it fails, the next one is tried
and the response records the fallback. You can cap the ladder per request
(`max_tier: "LOCAL"` gives you a strictly offline translation or an error) or
force one engine (`prefer: "google_free"`) to compare engines on one sentence.

---

## What's verified, and on what

Everything below was measured on the target machine (Ryzen 7 6800H, 16 GB RAM,
RTX 3050 Laptop 4 GB, Windows 11, Python 3.12) rather than assumed.

| Claim | Status |
|---|---|
| `IndicTransToolkit` builds on Windows | ✅ builds a `cp312-win_amd64` wheel |
| Local translation on the RTX 3050 | ✅ NLLB-600M in fp16, **0.19–0.46 s/sentence**, ~1.2 GB VRAM |
| Speech round trip (TTS → Whisper → translate) | ✅ Hindi exact; Tamil imperfect on `small` |
| Google gtx endpoint reachable | ✅ — **but blocks plain Python by TLS fingerprint**, see below |
| Google Input Tools transliteration | ✅ `mera naam rahul` → `मेरा नाम राहुल` |
| `ctranslate2` on Windows **and** linux-aarch64 | ✅ wheels for both, so speech works on laptop and Oracle |
| Edge TTS keyless endpoint | ✅ 322 voices, 35 Indian |
| Self-test suite | ✅ 55/55 with local models |
| English pivot beats NLLB direct | ✅ `hi→ja` 私の名前はラहुル → 私の名前はラウル |
| Wrong-script output is caught, not served | ✅ falls through to an engine that gets it right |
| Languages sharing a script are told apart | ✅ 11/11, was 2/11 — Marathi no longer read as Hindi |
| Cache | ✅ repeat request 190 ms → **0 ms**, survives restarts |
| Document translation | ✅ .txt .md .csv .srt .vtt .docx .pdf, structure preserved |
| MMS-TTS offline voices | ✅ Assamese, Odia, Maithili, Punjabi, all 36M params |
| **IndicTrans2 + IndicConformer weights** | ⚠️ **gated on Hugging Face — needs a free account**, see below |
| **transformers 5.x** | ⚠️ **breaks IndicTrans2**; pinned to `<5`, see below |

### IndicTrans2 is gated (one-time setup)

The weights are still MIT licensed and approval is automatic, but the Hub
marks them `gated: auto`, so an anonymous download gets a 401. Once:

1. Create a free account at <https://huggingface.co/join>.
2. Open each page and click **Agree and access repository**:
   [en-indic](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) ·
   [indic-en](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) ·
   [indic-indic](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M) ·
   [indic-conformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
3. Make a read token at <https://huggingface.co/settings/tokens>.
4. `set HF_TOKEN=hf_xxx` (or run `huggingface-cli login`).

Run `python scripts/check_setup.py` to see exactly which of these your machine
can reach.

Until you do, the app still works — the router logs the reason once, marks the
gated model unavailable, and falls through. NLLB, Whisper and MMS-TTS are all
ungated and need none of this. What you lose without the token: the strongest
Indic translation quality, and speech-to-text for the eight languages only
IndicConformer covers.

### transformers must stay on 4.x

transformers 5 moved `PreTrainedTokenizerBase` out of
`transformers.tokenization_utils`. Both `IndicTransToolkit` 1.1.1 and
IndicTrans2's own `trust_remote_code` modelling file still import it from
there, so on transformers 5 IndicTrans2 fails at *import*, before any
download. `requirements-local.txt` pins `transformers>=4.44,<5`; verified
working on **4.57.6**. NLLB itself is fine on either.

### The TLS fingerprint problem

Google's free endpoint returns **HTTP 429 to `httpx` and `requests` even with
browser-identical headers**, while `curl` with the same headers gets 200. It
fingerprints the TLS handshake, not the headers. Header spoofing does not fix
it; this is the single thing most tutorials get wrong.

The fix is `curl_cffi`, which presents a real browser handshake. It has wheels
for `win_amd64` and `manylinux aarch64`, so it works on the laptop and on the
Oracle ARM box. There is also a second, independent path
(`clients5.google.com`) that plain `httpx` *can* reach, kept as a fallback for
environments where `curl_cffi` won't install.

---

## Coverage

All 61 languages translate. Speech is narrower, because the models simply do
not exist for some languages — every number below was measured against the
live services and model hub, not assumed.

| | Total | Indian (of 23) | Foreign (of 38) |
|---|---|---|---|
| Translation | **61** | 23 | 38 |
| Speech-to-text | **60** | **23** | 37 |
| Text-to-speech | **53** | **15** | 38 |
| …of which work offline | 26 | 13 | 13 |

**Speech-to-text** chains two engines. Whisper (faster-whisper) is the only
one that can *identify* an unknown language, so it always runs first on
auto-detect. [IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
(MIT) covers all 22 scheduled languages — including the eight Whisper cannot
transcribe at all — and is more accurate on the rest. When Whisper detects an
Indian language, the audio is re-transcribed with IndicConformer and that text
wins; set `LINGUA_ASR_REFINE=0` to skip the second pass.

Only Zulu has no speech-to-text, because Whisper has no Zulu model.

**Text-to-speech** chains three. Edge (keyless, neural, best quality) covers
11 Indian languages; Google's keyless endpoint adds Punjabi; MMS-TTS (VITS,
36M params, ~150 MB each) runs **fully offline** and is the only voice that
exists for Assamese, Odia and Maithili — plus Hausa and Yoruba, the two
foreign languages Edge misses.

Unlike translation, speech defaults to quality-first rather than local-first,
because MMS is audibly more robotic than a neural voice. `LINGUA_TTS_ORDER`
changes that, and the UI's *Voice source → On-device only* forces MMS.

Eight languages still have no voice **anywhere public** — Bodo, Dogri,
Konkani, Kashmiri, Manipuri, Sanskrit, Santali, Sindhi. They translate and
(with IndicConformer) transcribe; they just cannot be spoken back. That is a
gap in what has been built in the world, not in this app.

---

## Accuracy checking

The brief this answers says accuracy is what decides whether a translation
service is worth using. So the system can estimate its own.

Ask for it per request (`"score": true`, or *Accuracy check* in the UI) and
the translation is fed back through a **different** engine, with the round
trip compared to the original using chrF — an F-score over character n-grams,
chosen because it needs no tokeniser, which matters when the languages include
Japanese and Tamil.

`"best_of": 3` runs three engines and keeps the highest scorer. One grader
scores every candidate: grading each with a different engine is not a
comparison, and an early version produced 1.0, 1.0 and 0.711 for three
engines that had returned *identical text*.

Read the numbers for what they are:

```
1.00  identical round trip
0.66  "My name is Rahul, I come from Delhi"     good paraphrase
0.59  "My boat is Rahul and I am from Mumbai"   WRONG, but still scores fine
0.31  "I am Rahul from Delhi"                   terse but correct
0.08  "The weather is nice today"               unrelated
```

Rows three and four are the honest limitation: this is surface similarity, so
one wrong content word costs little while a correct-but-terse rendering is
punished. It is a smoke alarm for gross failure and a fair tie-breaker between
candidates — not a judge of meaning. That is why the back translation is
displayed next to the score: a person reading "My boat is Rahul" needs no
metric.

One blind spot is closed explicitly. A provider that hands the text back
untranslated round-trips *perfectly*, and scored 0.61 — enough to beat real
translations. Output is therefore also compared against the **input**, and a
passthrough is forced to zero.

---

## Documents

```bash
curl -X POST localhost:8000/api/document/translate \
  -F "file=@notice.docx" -F "target=hi" -o notice.hi.docx
```

| In | Out | What survives |
|---|---|---|
| `.txt` `.md` `.csv` `.log` | same | blank lines, indentation, markdown markers |
| `.srt` `.vtt` | same | timestamps and cue numbers are never sent to an engine |
| `.docx` | same | paragraph styles, bold runs, tables — written back into the original file |
| `.pdf` | `.docx` | text per page, paragraphs rebuilt from wrapped lines |

PDF is the one asymmetry, deliberately: rewriting a PDF in place means
re-flowing text into fixed-position boxes in a script whose glyphs are a
different width, which overlaps and clips. An editable Word file is more
honest than a broken-looking PDF.

Repeated segments are translated once — a subtitle file with the same line
twice costs one call, and `/api/document/inspect` reports the saving before
you commit. A segment that fails keeps its original text rather than losing
the document.

---

## Caching

On by default, in memory with a SQLite file behind it so it survives
restarts. A repeated sentence went from **190 ms to 0 ms** in testing.

It matters for more than speed: Google's keyless endpoints rate-limit
datacentre IPs far harder than home connections, which is the main fragility
of the Render deployment. Every hit is a call not made. Transliterations are
cached too, since the live input hint calls that on every pause in typing.

`LINGUA_CACHE=0` disables it; `use_cache: false` bypasses it for one request;
`POST /api/cache/clear` empties it.

---

## Install

Requires **Python 3.12** (not 3.13/3.14 — PyTorch support lags).

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux/macOS

pip install -r requirements-server.txt
```

That gives you a working app on the free network tiers, plus speech.

### Adding the local models

```bash
# NVIDIA GPU (RTX 3050, CUDA 12.x driver)
pip install torch --index-url https://download.pytorch.org/whl/cu126
# or CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-local.txt
python scripts/download_models.py           # ~5 GB, do it once, not at first request
```

On Windows `IndicTransToolkit` compiles from source and needs the MSVC build
tools (Visual Studio "Desktop development with C++"). Verified working on this
machine.

**Disk budget**, measured: the virtualenv is **~4.8 GB** once CUDA torch is in
(the torch wheel alone is 2.6 GB), plus roughly 5 GB of model weights in
`~/.cache/huggingface`. Budget ~10 GB. The CPU build of torch is far smaller
(~250 MB) if you do not need the GPU.

If `pip install torch` stalls partway — it did here, twice, sitting at
1.7 GB of 2.6 GB with no progress — pip cannot resume. Fetch the wheel with
something that can, then install the local file:

```bash
curl -L -C - --retry 20 --retry-all-errors -o torch.whl \
  https://download.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp312-cp312-win_amd64.whl
pip install torch.whl
```

### Run

```bash
.venv\Scripts\python.exe -m uvicorn server.main:app --reload --port 8000
```

Open <http://localhost:8000>. Use `localhost`, not the LAN IP — browsers only
grant microphone access on `localhost` or HTTPS.

### Check it

```bash
.venv\Scripts\python.exe scripts\check_setup.py         # what works, what to fix
.venv\Scripts\python.exe scripts\selftest.py            # everything
.venv\Scripts\python.exe scripts\selftest.py --offline  # local models only
```

`check_setup.py` is the one to run first: it reports missing packages, an
incompatible transformers, free VRAM, and exactly which gated models your
token can reach.

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `LINGUA_EDITION` | `full` | `full` or `lite` (no local models) |
| `LINGUA_LOCAL` | `1` | set `0` to skip local models entirely |
| `LINGUA_DEVICE` | `auto` | `cuda` / `cpu`. Auto refuses CUDA below 1.2 GB free VRAM |
| `LINGUA_MODEL_CACHE` | `2` | how many models stay resident before LRU eviction |
| `LINGUA_MAX_TIER` | `FREEMIUM` | ceiling on fallback |
| `LINGUA_LOCAL_TIMEOUT` | `120` | seconds before a local model gives up and falls through |
| `LINGUA_WHISPER_MODEL` | `small` | `tiny`…`large-v3` |
| `LINGUA_ASR_REFINE` | `1` | re-transcribe Indic audio with IndicConformer for accuracy |
| `LINGUA_CONFORMER_DECODING` | `ctc` | `ctc` or `rnnt` |
| `LINGUA_TTS_ORDER` | `edge,google,mms` | voice engine order; put `mms` first to prefer offline |
| `LINGUA_TTS_CACHE` | `2` | how many MMS voices stay resident |
| `LINGUA_CACHE` | `1` | `0` disables the translation cache |
| `LINGUA_CACHE_TTL` | 30 days | seconds before a cached entry expires |
| `LINGUA_CACHE_SIZE` | `512` | entries held in memory |
| `LINGUA_CACHE_DIR` | `~/.cache/lingua` | where the SQLite cache lives |
| `LINGUA_DOC_CONCURRENCY` | `4` | document segments translated in parallel |
| `LINGUA_MAX_DOC_MB` | `20` | upload limit for documents |
| `LINGUA_PRELOAD` | unset | `1` warms models at startup |
| `MYMEMORY_EMAIL` | unset | raises MyMemory's free quota ~5k → ~50k words/day |
| `LIBRETRANSLATE_URL` | unset | point at your own instance to get an unlimited free tier |

The 4 GB VRAM guard matters: `pick_device` measures free VRAM and drops to CPU
rather than dying with an OOM mid-request.

---

## API

```bash
curl -X POST localhost:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"mera naam rahul hai","source":"auto","target":"en"}'
```

```json
{
  "text": "My name is Rahul",
  "source": "hi", "target": "en",
  "provider": "google_free", "tier": "FREE_WEB",
  "romanized_input": true,
  "native_input": "मेरा नाम राहुल है",
  "detected_by": "romanized-heuristic",
  "took_ms": 194,
  "notes": ["read as Hindi via google_inputtools"]
}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/translate` | text → text |
| `POST /api/transliterate` | romanized → native script, on its own |
| `POST /api/speech/transcribe` | audio → text |
| `POST /api/speech/translate` | audio → transcript + translation |
| `POST /api/speak` | text → mp3 or wav |
| `POST /api/document/inspect` | what's in a file, before translating it |
| `POST /api/document/translate` | file in, translated file out |
| `GET /api/document/formats` | accepted formats and size limit |
| `POST /api/cache/clear` | empty the translation cache |
| `GET /api/languages` | the picker payload |
| `GET /api/providers` | which engines are live, on what device |

---

## Deploying

### Oracle Cloud — the full app, free, always on

Always Free ARM Ampere. **Note the June 2026 change: the allocation is now
2 OCPU / 12 GB, halved from 4 / 24.** Still plenty for IndicTrans2 + Whisper
on CPU.

```bash
bash scripts/deploy_oracle.sh
```

Installs CPU torch, pre-downloads weights, writes a systemd unit, opens the
port. If the public URL times out afterwards it is almost always the **VCN
security list** in the OCI console, not the instance firewall — the script
handles the latter but cannot touch the former.

### Render / Vercel — the lite edition

Free Render is **512 MB RAM and 0.1 CPU and sleeps after 15 minutes**. No
local translation model loads in that, so `lite/` drops tier 0 and serves the
same UI over the network tiers only.

- **Render**: Blueprint → `lite/render.yaml`. Root stays the repo root so the
  `lingua` package is importable.
- **Vercel**: `vercel.json` and `api/index.py` at the repo root (Vercel
  requires functions in `/api`); they are three-line shims onto `lite/main.py`.

Honest caveat: Google's keyless endpoints rate-limit datacentre IPs far more
aggressively than home connections. The ladder falls through to MyMemory so
the app keeps working, but quality drops. Set `LIBRETRANSLATE_URL` to an
instance you control if that matters.

### What about Hugging Face Spaces?

Checked, and it changed: **Gradio and Docker Spaces now require a paid plan.**
The one free route is 2 ZeroGPU Gradio Spaces per personal account (verified
email, >30 days old), which would mean rewriting the UI in Gradio and living
with a 5 GPU-minute daily quota per viewer. Good for a flashy demo, poor as a
product, so it isn't wired up here.

---

## Licensing

Relevant if this goes past a college project:

| Component | Licence | Commercial? |
|---|---|---|
| IndicTrans2 | MIT | ✅ |
| Whisper / faster-whisper | MIT / Apache-2.0 | ✅ |
| `indic-transliteration` | MIT | ✅ |
| **NLLB-200** | **CC-BY-NC-4.0** | ❌ **non-commercial only** |
| Google gtx / Input Tools / Edge TTS | undocumented endpoints | ❌ no agreement covers this |

To make it commercially clean: drop NLLB for MADLAD-400 (Apache-2.0) or
`Helsinki-NLP/opus-mt-*`, cap `LINGUA_MAX_TIER=LOCAL`, and self-host
LibreTranslate. Nothing else changes — that is what the provider abstraction
is for.

IndicTrans2 loads with `trust_remote_code=True`, which executes code from the
model repo. They are first-party AI4Bharat repos, but you should know it.

---

## How it's put together

```
lingua/                  provider-agnostic engine, no web framework
  languages.py           61 languages ×  the 5 code systems providers use
  script.py              Unicode script detection + romanized-Indic heuristic
  router.py              tier ladder, source resolution, transliteration fork
  cache.py               memory LRU over a SQLite file
  quality.py             chrF, round-trip scoring, passthrough detection
  providers/
    google_free.py       gtx (curl_cffi) + clients5 (httpx)
    inputtools.py        romanized → native, online
    xlit_local.py        romanized → native, offline, approximate
    indictrans2.py       local, Indian languages
    nllb.py              local, everything else
    pivot.py             two hops via English for Indic ↔ foreign
    free_apis.py         MyMemory, LibreTranslate
server/                  full edition: FastAPI + speech + the UI
  asr.py / asr_engines.py    Whisper + IndicConformer chain
  tts.py / tts_engines.py    Edge + Google + MMS chain
  documents.py               .txt .md .srt .vtt .docx .pdf
lite/                    network-only edition for Render/Vercel
scripts/                 selftest, model download, Oracle provisioning
```

The engine knows nothing about HTTP; both editions are thin shells over the
same `Router`. Adding a provider is one class and one line in `engine.py`.

### Telling apart languages that share a script

Devanagari carries Hindi, Marathi, Nepali, Sanskrit, Maithili, Bodo, Dogri and
Konkani. Bengali script carries Bengali and Assamese. Arabic script carries
Urdu, Sindhi and Kashmiri. Assuming the most common one is not a rounding
error — it changes meaning:

```
माझे नाव राहुल आहे   read as Hindi    →  "My boat is Rahul"
                     read as Marathi  →  "My name is Rahul"
```

…because नाव is *boat* in Hindi and *name* in Marathi. Before this, 5 of 6
Devanagari languages were silently treated as Hindi, and a threshold bug
(`confidence < 0.6` against a value of exactly `0.6`) meant even the online
path never asked Google, which detects them correctly.

`script.disambiguate_script` now decides, using two signals:

- **Signature characters**, which are decisive. Assamese writes `ৰ` and `ৱ`
  where Bengali writes `র` and `ব`; Sindhi and Kashmiri each have letters Urdu
  never uses.
- **Function-word lexicons** otherwise — copulas and pronouns, because they
  appear in nearly every sentence and rarely cross over: Hindi `है`, Marathi
  `आहे`, Nepali `छ`, Maithili `अछि`, Sanskrit `अस्ति`.

It reports honest confidence, and below `router.DETECT_THRESHOLD` the router
defers to a provider that can detect for itself. 11 of 11 languages are now
identified correctly, offline, with no extra dependency.

### The output sanity check

Small models garble proper nouns across scripts. NLLB-600M renders "Rahul"
into Japanese as `ラहुल` — Devanagari spliced into katakana — and it does this
*even when handed clean English input*, so no amount of routing fixes it.

`script.looks_like_wrong_script` catches it: a translation whose target script
is Japanese should not contain Devanagari. Latin is always allowed (names,
acronyms and loanwords appear in Latin letters in every language), Japanese
may mix kana with Han, Korean may mix Hangul with Han, and anything below 2%
of characters is treated as incidental.

When it fires, the router records it as a provider failure and tries the next
one:

```
hi→ja, LOCAL only    私の名前はラहुल…        served, flagged "best available, but
                                            Devanagari characters in Japanese output"
hi→ja, network on    私の名前はラーフル、デリー出身です   falls through to Google, correct katakana
```

If every provider fails the check, the first result is returned anyway with
the reason attached — a garbled translation still beats no translation.

### Two decisions worth reading the code for

**Source-language resolution** (`router.resolve_source`) happens *before*
choosing a provider, because local NMT models cannot detect language at all
and would otherwise be unusable with "Detect language" selected. Script
detection works offline; when confidence is low and we're going out to the
network anyway, the provider's own detector is trusted instead.

**The romanized fork** (`router.translate`) sends native script to everyone
once a trustworthy transliteration exists — including Google, per the table at
the top. The exception is the offline transliterator, which is marked
`approximate` and only fed to local models, since Google's own Hinglish
handling beats a rule-based guess.

### Known limits

- Offline romanized input is approximate: a ~200-word Hindi lexicon plus
  rule-based fallback. Casual romanization drops vowel length (`mera` vs
  `meraa`) and no rule system recovers that reliably. Online it's exact.
- Devanagari is shared by Hindi, Marathi, Nepali, Sanskrit, Konkani and
  Maithili, so script alone gives 0.6 confidence and defaults to Hindi. Pick
  the language explicitly when it matters.
- **Speech still covers fewer languages than text** — see
  [Coverage](#coverage) for the current numbers and which eight languages
  cannot be spoken back.
- **Whisper and IndicConformer disagree about what is possible.** Asking for a
  language Whisper has no model for (Odia, say) now *fails* rather than
  falling back to Whisper, because Whisper would cheerfully transcribe the
  audio in some other language and label it Odia. If you see that error, the
  fix is the HF token.
- **Whisper `small` is weak on Indic speech.** A measured round trip
  (synthesise → transcribe → translate) came back perfect for Hindi but
  garbled Tamil: `இன்று` heard as `இந்து` and `அழகாக` as `அறகாக`, which turned
  "the weather today is beautiful" into "The Hindu sky is very peaceful".
  With IndicConformer enabled this is largely fixed for Indian languages; for
  everything else set `LINGUA_WHISPER_MODEL=medium` (~1.5 GB). On CPU, `small`
  runs about 1.6× real time — a 3-second clip took 4.8 s.
