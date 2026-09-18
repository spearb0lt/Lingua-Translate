/* Lingua front end. Vanilla JS, no build step -- it has to run from a
   512 MB Render instance and an ARM box with nothing installed. */

const $ = (id) => document.getElementById(id);

const el = {
  srcLang: $("src-lang"), tgtLang: $("tgt-lang"),
  srcText: $("src-text"), out: $("out"),
  hint: $("hint"), hintNative: $("hint-native"),
  hintTag: $("hint-tag"), hintApprox: $("hint-approx"),
  go: $("go"), swap: $("swap"), copy: $("copy"), clear: $("clear"),
  share: $("share"),
  mic: $("mic"), micLabel: $("mic-label"), micGlyph: $("mic-glyph"),
  speak: $("speak"), count: $("count"), status: $("status"),
  enginePill: $("engine-pill"), engineText: $("engine-text"),
  maxTier: $("max-tier"), prefer: $("prefer"), voiceGender: $("voice-gender"),
  voiceSource: $("voice-source"), qualityMode: $("quality-mode"),
  docFile: $("doc-file"), docGo: $("doc-go"), docInspect: $("doc-inspect"),
  docStatus: $("doc-status"), docFormats: $("doc-formats"),
  provTable: $("prov-table").querySelector("tbody"),
  toast: $("toast"), player: $("player"),
};

let languages = [];           // flat list of language rows
let lastResult = null;
let inFlight = null;          // AbortController for the current translate
let ttsSilent = new Set();    // language names with no voice on any engine

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

function toast(message) {
  el.toast.textContent = message;
  el.toast.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.toast.classList.remove("show"), 2200);
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res;
}

const json = (path, body, signal) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
  signal,
}).then((r) => r.json());

/* ------------------------------------------------------------------ */
/* startup                                                             */
/* ------------------------------------------------------------------ */

async function loadLanguages() {
  const { groups } = await (await api("/api/languages")).json();

  for (const target of [el.srcLang, el.tgtLang]) {
    for (const group of groups) {
      const og = document.createElement("optgroup");
      og.label = group.group;
      for (const item of group.items) {
        const opt = document.createElement("option");
        opt.value = item.code;
        opt.textContent = item.label;
        og.appendChild(opt);
      }
      target.appendChild(og);
    }
  }
  languages = groups.flatMap((g) => g.items);

  el.srcLang.value = localStorage.getItem("lingua.src") || "auto";
  el.tgtLang.value = localStorage.getItem("lingua.tgt") || "en";
  el.maxTier.value = localStorage.getItem("lingua.tier") || "FREE_WEB";
  el.voiceSource.value = localStorage.getItem("lingua.voiceSource") || "auto";
  el.qualityMode.value = localStorage.getItem("lingua.quality") || "off";
}

async function loadProviders() {
  let data;
  try {
    data = await (await api("/api/providers")).json();
  } catch (err) {
    setEngine("offline", `API unreachable: ${err.message}`, true);
    return;
  }

  el.provTable.innerHTML = "";
  el.prefer.length = 1;

  for (const p of data.translation) {
    const row = document.createElement("tr");
    const ready = p.available;
    const notes = [p.blurb];
    if (p.status?.device) notes.push(`device: ${p.status.device}`);
    if (p.status?.licence) notes.push(p.status.licence);

    row.innerHTML = `
      <td class="name">${p.name}</td>
      <td>${p.tier}</td>
      <td class="${ready ? "yes" : "no"}">${ready ? "yes" : "no"}</td>
      <td>${notes.filter(Boolean).join(" · ")}</td>`;
    el.provTable.appendChild(row);

    if (ready) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = `${p.name} (${p.tier})`;
      el.prefer.appendChild(opt);
    }
  }

  for (const [label, info] of [["speech-to-text", data.speech_to_text],
                               ["text-to-speech", data.text_to_speech]]) {
    const row = document.createElement("tr");
    const notes = [];
    if (info.languages) {
      notes.push(`${info.languages} languages, ${info.indian_languages} Indian`);
    }
    if (info.model) notes.push(`model: ${info.model}`);
    if (info.device) notes.push(`device: ${info.device}`);
    if (info.engines) {
      const live = info.engines.filter((e) => e.available).map((e) => e.name);
      if (live.length) notes.push(`engines: ${live.join(" + ")}`);
    }
    if (info.offline_languages) notes.push(`${info.offline_languages} offline`);
    if (info.note) notes.push(info.note);
    row.innerHTML = `
      <td class="name">${label}</td><td>—</td>
      <td class="${info.available ? "yes" : "no"}">${info.available ? "yes" : "no"}</td>
      <td>${notes.join(" · ") || "not installed"}</td>`;
    el.provTable.appendChild(row);
  }

  if (data.cache) {
    const c = data.cache;
    const row = document.createElement("tr");
    const rate = c.hit_rate !== null && c.hit_rate !== undefined
      ? ` \u00b7 ${Math.round(c.hit_rate * 100)}% hit rate` : "";
    row.innerHTML = `
      <td class="name">cache</td><td>\u2014</td>
      <td class="${c.enabled ? "yes" : "no"}">${c.enabled ? "on" : "off"}</td>
      <td>${c.memory_entries} in memory, ${c.disk_entries || 0} on disk${rate}</td>`;
    el.provTable.appendChild(row);
  }
  if (data.documents) {
    el.docFormats.textContent =
      `Accepts ${data.documents.formats.join(", ")} up to ${data.documents.max_mb} MB. ` +
      `Keeps subtitle timings, Word styling and tables, and indentation. ` +
      `PDFs come back as editable Word files.`;
  }

  const localReady = data.translation.some((p) => p.tier === "LOCAL" && p.available);
  const anyReady = data.translation.some((p) => p.available);
  if (localReady) setEngine("LOCAL", "local models ready");
  else if (anyReady) setEngine("FREE_WEB", "network engines ready");
  else setEngine("offline", "no engine available", true);

  el.mic.disabled = !data.speech_to_text.available;
  if (!data.speech_to_text.available) {
    el.mic.title = "Speech recognition not installed (requirements-local.txt)";
  }
  el.speak.disabled = !data.text_to_speech.available;
  ttsSilent = new Set(data.text_to_speech.silent || []);
  updateSpeakButton();
}

function updateSpeakButton() {
  const lang = languages.find((l) => l.code === el.tgtLang.value);
  const silent = lang && ttsSilent.has(lang.name);
  el.speak.disabled = silent || !lang;
  el.speak.title = silent
    ? `No voice exists for ${lang.name} on any engine`
    : "Read the translation aloud";
}

function setEngine(tier, text, isError = false) {
  el.enginePill.className = "pill" + (isError ? " err" : ` tier-${tier}`);
  el.engineText.textContent = text;
}

/* ------------------------------------------------------------------ */
/* the romanization hint                                               */
/* ------------------------------------------------------------------ */

const HINT_LANGS = new Set(["hi", "bn", "ta", "te", "mr", "gu", "kn",
                            "ml", "pa", "ur", "ne", "sa", "as", "or"]);

function looksRomanized(text) {
  // Cheap client-side gate so we do not hit the API on every keystroke.
  // The server decides for real; this just avoids obvious no-ops.
  if (!/[a-zA-Z]/.test(text)) return false;
  if (/[ऀ-෿؀-ۿ]/.test(text)) return false;
  return text.trim().split(/\s+/).length >= 2;
}

const updateHint = debounce(async () => {
  const text = el.srcText.value.trim();
  const src = el.srcLang.value;

  if (!text || !looksRomanized(text)) return hideHint();
  // Only ask when the user has actually picked an Indic language; for
  // "Detect language" we let the translate call surface it instead, so we
  // are not transliterating every English sentence someone types.
  if (!HINT_LANGS.has(src)) return hideHint();

  try {
    const data = await json("/api/transliterate", { text, language: src });
    if (data.text && data.text !== text) {
      el.hintNative.textContent = data.text;
      el.hintApprox.textContent =
        data.provider === "local_xlit" ? "(offline, approximate)" : "";
      el.hint.classList.add("show");
    } else hideHint();
  } catch { hideHint(); }
}, 420);

function hideHint() { el.hint.classList.remove("show"); }

function showHintFromResult(result) {
  if (result.native_input && result.romanized_input) {
    const lang = languages.find((l) => l.code === result.source);
    el.hintTag.textContent = lang ? `read as ${lang.name}` : "read as";
    el.hintNative.textContent = result.native_input;
    const note = (result.notes || []).find((n) => n.includes("approximate"));
    el.hintApprox.textContent = note ? "(offline, approximate)" : "";
    el.hint.classList.add("show");
  } else hideHint();
}

/* ------------------------------------------------------------------ */
/* translate                                                           */
/* ------------------------------------------------------------------ */

async function translate() {
  const text = el.srcText.value.trim();
  if (!text) { toast("Nothing to translate"); return; }

  if (inFlight) inFlight.abort();
  inFlight = new AbortController();

  el.go.disabled = true;
  el.go.textContent = "Translating…";
  el.status.innerHTML = "";

  const started = performance.now();
  try {
    const mode = el.qualityMode.value;
    const result = await json("/api/translate", {
      text,
      source: el.srcLang.value,
      target: el.tgtLang.value,
      max_tier: el.maxTier.value,
      prefer: el.prefer.value || null,
      score: mode !== "off",
      best_of: mode === "best" ? 3 : 1,
    }, inFlight.signal);

    lastResult = result;
    el.out.textContent = result.text;
    el.out.classList.remove("empty");
    showHintFromResult(result);
    renderStatus(result, performance.now() - started);
    setEngine(result.tier, `${result.provider} · ${result.tier}`);
  } catch (err) {
    if (err.name === "AbortError") return;
    el.out.textContent = "";
    el.out.classList.add("empty");
    el.status.innerHTML = `<span class="pill err"><span class="dot"></span>${err.message}</span>`;
  } finally {
    el.go.disabled = false;
    el.go.textContent = "Translate";
    inFlight = null;
  }
}

function renderStatus(result, wallMs) {
  const srcName = languages.find((l) => l.code === result.source)?.name || result.source;
  const bits = [
    `<span class="pill tier-${result.tier}"><span class="dot"></span>${result.provider}</span>`,
    `<span>${srcName} → ${languages.find((l) => l.code === result.target)?.name || result.target}</span>`,
    `<span class="mono">${result.took_ms} ms server · ${Math.round(wallMs)} ms total</span>`,
  ];
  if (result.cached) bits.push(`<span class="note">from cache</span>`);
  if (result.detected_by && result.detected_by !== "explicit") {
    bits.push(`<span class="note">detected by ${result.detected_by}</span>`);
  }
  if (result.quality !== null && result.quality !== undefined) {
    const band = qualityBand(result.quality);
    bits.push(`<span>accuracy <span class="quality ${band}">${Math.round(result.quality * 100)}% ${band}</span></span>`);
  }
  for (const note of result.notes || []) bits.push(`<span class="note">${note}</span>`);
  el.status.innerHTML = bits.join("");

  if (result.back_translation) {
    const div = document.createElement("div");
    div.className = "back-translation";
    div.style.flexBasis = "100%";
    div.textContent = `translated back by ${result.graded_by}: "${result.back_translation}"`;
    el.status.appendChild(div);
  }
  if (result.alternatives && result.alternatives.length) {
    const box = document.createElement("div");
    box.className = "alts";
    box.style.flexBasis = "100%";
    for (const alt of result.alternatives) {
      const row = document.createElement("div");
      row.className = "alt";
      const pct = (alt.quality === null || alt.quality === undefined)
        ? "\u2014" : `${Math.round(alt.quality * 100)}%`;
      row.innerHTML = `<b>${escapeHtml(alt.provider)}</b> ${pct}: ${escapeHtml(alt.text)}`;
      box.appendChild(row);
    }
    el.status.appendChild(box);
  }
}

function qualityBand(score) {
  if (score >= 0.75) return "high";
  if (score >= 0.5) return "moderate";
  if (score >= 0.3) return "low";
  return "very";
}

function escapeHtml(text) {
  const node = document.createElement("span");
  node.textContent = text;
  return node.innerHTML;
}

/* ------------------------------------------------------------------ */
/* documents                                                           */
/* ------------------------------------------------------------------ */

async function inspectDocument() {
  const file = el.docFile.files && el.docFile.files[0];
  if (!file) { toast("Choose a file first"); return; }
  el.docStatus.textContent = "reading\u2026";
  const form = new FormData();
  form.append("file", file);
  try {
    const info = await (await api("/api/document/inspect",
                                  { method: "POST", body: form })).json();
    const unique = info.unique_segments < info.segments
      ? ` (${info.unique_segments} unique)` : "";
    el.docStatus.textContent =
      `${info.kind}: ${info.segments} segments${unique}, ${info.characters} chars`;
    if (info.note) toast(info.note);
  } catch (err) {
    el.docStatus.textContent = err.message;
  }
}

async function translateDocument() {
  const file = el.docFile.files && el.docFile.files[0];
  if (!file) { toast("Choose a file first"); return; }

  el.docGo.disabled = true;
  el.docStatus.textContent = "translating\u2026";
  const form = new FormData();
  form.append("file", file);
  form.append("target", el.tgtLang.value);
  form.append("source", el.srcLang.value);
  form.append("max_tier", el.maxTier.value);

  try {
    const res = await api("/api/document/translate", { method: "POST", body: form });
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const name = match ? match[1] : `translated-${file.name}`;

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);

    const failed = Number(res.headers.get("X-Failed-Segments") || 0);
    el.docStatus.textContent =
      `${res.headers.get("X-Segments")} segments \u00b7 ` +
      `${res.headers.get("X-Source-Language")} \u2192 ${el.tgtLang.value}` +
      (failed ? ` \u00b7 ${failed} kept as original` : "") +
      ` \u00b7 saved as ${name}`;
    const note = res.headers.get("X-Note");
    if (note) toast(note);
  } catch (err) {
    el.docStatus.textContent = err.message;
  } finally {
    el.docGo.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* speech in                                                           */
/* ------------------------------------------------------------------ */

let recorder = null;
let chunks = [];

async function toggleRecording() {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }

  if (!navigator.mediaDevices?.getUserMedia) {
    toast("Microphone needs HTTPS or localhost");
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    toast("Microphone permission denied");
    return;
  }

  chunks = [];
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"]
    .find((m) => MediaRecorder.isTypeSupported(m)) || "";
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);

  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    setRecordingUI(false);
    const blob = new Blob(chunks, { type: mime || "audio/webm" });
    if (blob.size < 1200) { toast("Recording too short"); return; }
    await sendAudio(blob, mime);
  };

  recorder.start();
  setRecordingUI(true);
}

function setRecordingUI(recording) {
  el.mic.classList.toggle("rec", recording);
  el.micGlyph.textContent = recording ? "" : "🎙";
  el.micGlyph.className = recording ? "dot" : "";
  el.micLabel.textContent = recording ? "Stop" : "Speak";
}

async function sendAudio(blob, mime) {
  const ext = mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("audio", blob, `clip.${ext}`);
  form.append("target", el.tgtLang.value);
  form.append("source", el.srcLang.value);
  form.append("max_tier", el.maxTier.value);

  el.status.innerHTML = `<span class="pill"><span class="dot"></span>transcribing…</span>`;
  try {
    const data = await (await api("/api/speech/translate", { method: "POST", body: form })).json();

    el.srcText.value = data.transcript.text;
    el.count.textContent = data.transcript.text.length;

    if (!data.translation) {
      el.status.innerHTML = `<span class="pill err"><span class="dot"></span>${data.note || "no speech detected"}</span>`;
      return;
    }

    lastResult = data.translation;
    el.out.textContent = data.translation.text;
    el.out.classList.remove("empty");
    showHintFromResult(data.translation);
    renderStatus(data.translation, 0);

    const heard = languages.find((l) => l.code === data.transcript.language)?.name
                  || data.transcript.language;
    el.status.insertAdjacentHTML("beforeend",
      `<span class="note">heard ${heard} · ${data.transcript.provider} · ${data.transcript.took_ms} ms</span>`);

    if (el.speak.dataset.auto === "1") await speakOutput();
  } catch (err) {
    el.status.innerHTML = `<span class="pill err"><span class="dot"></span>${err.message}</span>`;
  }
}

/* ------------------------------------------------------------------ */
/* speech out                                                          */
/* ------------------------------------------------------------------ */

async function speakOutput() {
  const text = lastResult?.text || el.out.textContent;
  if (!text || el.out.classList.contains("empty")) { toast("Nothing to read"); return; }

  el.speak.disabled = true;
  try {
    const res = await api("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: el.tgtLang.value,
        gender: el.voiceGender.value || null,
        offline_only: el.voiceSource.value === "offline",
      }),
    });
    const buf = await res.blob();
    el.player.src = URL.createObjectURL(buf);
    await el.player.play();
    const voice = res.headers.get("X-Voice");
    const engine = res.headers.get("X-TTS-Engine");
    const offline = res.headers.get("X-TTS-Offline") === "true";
    if (voice) {
      toast(`${voice} · ${engine}${offline ? " · offline" : ""}`);
    }
  } catch (err) {
    toast(err.message);
  } finally {
    el.speak.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* wiring                                                              */
/* ------------------------------------------------------------------ */

el.go.addEventListener("click", translate);
el.docGo.addEventListener("click", translateDocument);
el.docInspect.addEventListener("click", inspectDocument);
el.docFile.addEventListener("change", () => { el.docStatus.textContent = ""; });
el.qualityMode.addEventListener("change", () =>
  localStorage.setItem("lingua.quality", el.qualityMode.value));
el.speak.addEventListener("click", speakOutput);
el.mic.addEventListener("click", toggleRecording);

el.clear.addEventListener("click", () => {
  el.srcText.value = "";
  el.count.textContent = "0";
  el.out.textContent = "Translation appears here.";
  el.out.classList.add("empty");
  el.status.innerHTML = "";
  hideHint();
  el.srcText.focus();
});

el.copy.addEventListener("click", async () => {
  const text = lastResult?.text || "";
  if (!text) { toast("Nothing to copy"); return; }
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { toast("Copy failed"); }
});

el.share.addEventListener("click", async () => {
  if (!el.srcText.value.trim()) { toast("Nothing to link to"); return; }
  try { await navigator.clipboard.writeText(shareUrl()); toast("Link copied"); }
  catch { toast("Copy failed"); }
});

el.swap.addEventListener("click", () => {
  const from = el.srcLang.value, to = el.tgtLang.value;
  if (from === "auto") {
    if (!lastResult) { toast("Translate once first, then swap"); return; }
    el.srcLang.value = lastResult.target;
    el.tgtLang.value = lastResult.source;
  } else {
    el.srcLang.value = to;
    el.tgtLang.value = from;
  }
  if (!el.out.classList.contains("empty")) {
    el.srcText.value = el.out.textContent;
    el.count.textContent = el.srcText.value.length;
    el.out.textContent = "Translation appears here.";
    el.out.classList.add("empty");
  }
  persist();
  hideHint();
});

el.srcText.addEventListener("input", () => {
  el.count.textContent = el.srcText.value.length;
  updateHint();
});

el.srcText.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); translate(); }
});

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "m") {
    e.preventDefault(); toggleRecording();
  }
});

function persist() {
  localStorage.setItem("lingua.src", el.srcLang.value);
  localStorage.setItem("lingua.tgt", el.tgtLang.value);
  localStorage.setItem("lingua.tier", el.maxTier.value);
}
for (const node of [el.srcLang, el.tgtLang, el.maxTier]) {
  node.addEventListener("change", () => {
    persist(); hideHint(); updateHint(); updateSpeakButton();
  });
}
el.voiceSource.addEventListener("change", () =>
  localStorage.setItem("lingua.voiceSource", el.voiceSource.value));

/* Deep links. Accepts the same parameter names Google Translate uses, so a
   URL like  ?sl=as&tl=en&text=amar%20nam%20gobindo  works here unchanged. */
function applyUrlParams() {
  const q = new URLSearchParams(location.search);
  const pick = (...names) => names.map((n) => q.get(n)).find((v) => v);

  const known = (code) => code && languages.some((l) => l.code === code);

  const from = pick("sl", "from", "source");
  const to = pick("tl", "to", "target");
  const text = pick("text", "q");

  if (from === "auto" || known(from)) el.srcLang.value = from;
  if (known(to)) el.tgtLang.value = to;

  // ?score=1 or ?score=best turns the accuracy check on for a shared link,
  // so a link can demonstrate the scoring rather than just the translation.
  const score = q.get("score");
  if (score === "best") el.qualityMode.value = "best";
  else if (score === "1" || score === "true") el.qualityMode.value = "score";
  if (text) {
    el.srcText.value = text;
    el.count.textContent = text.length;
  }
  return Boolean(text);
}

function shareUrl() {
  const q = new URLSearchParams({
    sl: el.srcLang.value, tl: el.tgtLang.value, text: el.srcText.value,
  });
  return `${location.origin}${location.pathname}?${q}`;
}

(async function boot() {
  try {
    await loadLanguages();
    const shouldTranslate = applyUrlParams();
    await loadProviders();
    if (shouldTranslate) await translate();
  } catch (err) {
    setEngine("offline", `startup failed: ${err.message}`, true);
  }
})();
