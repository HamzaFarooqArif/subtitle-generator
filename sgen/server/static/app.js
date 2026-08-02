"use strict";

/* ===================================================================== state */

const state = {
  meta: null,
  cwd: null,
  selection: new Map(),   // path -> {name, size}
  jobs: new Map(),        // id -> job
  library: [],            // transcripts on disk, survives server restarts
  providers: null,        // which online translators have keys
  scan: null,             // last folder scan: what still needs doing
  excluded: new Set(),    // folder-mode: unfinished files the user unticked
  included: new Set(),    // folder-mode: finished files the user asked to redo
  fileSettings: null,     // path of the file open on the Settings panel's 2nd tab
  fileDirty: false,       // …and whether it has edits that are not written yet
  translate: { contentId: null, name: "" },
  tune: { contentId: null, values: {}, timer: null },
};

/* =================================================================== helpers */

/**
 * Element lookup that degrades instead of exploding.
 *
 * Listeners are attached at module scope, so one missing element used to throw
 * and kill the rest of the file — which took the file browser down with it even
 * though nothing was wrong with it. An inert detached node keeps a stale cache
 * or renamed id contained to the one control, and says so in the console.
 */
const $ = (sel) => {
  const el = document.querySelector(sel);
  if (el) return el;
  console.warn(`sgen: no element matches ${sel} — that control is inactive`);
  return document.createElement("div");
};
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/**
 * For elements that legitimately may not be there.
 *
 * `$` is deliberately never null, which makes it the wrong tool for asking
 * whether something exists: `!!$(sel)` is always true, and mutating what it
 * returns mutates a detached node that nobody can see.
 */
const find = (sel) => document.querySelector(sel);

/**
 * The backend is gone: stop pretending the page still works.
 *
 * Every control here depends on a server on this machine, so when that server
 * disappears — a restart, a crash, a killed terminal — the page becomes a
 * museum piece: it renders the last state it saw and silently drops every
 * click. This makes that state visible and offers the only real fix.
 */
function showOffline(reason = "") {
  $("#offline").classList.remove("hidden");
  $("#offline-reason").textContent = reason;
  $("#env").innerHTML = `<span class="dot bad"></span> not connected`;
}

function hideOffline() {
  $("#offline").classList.add("hidden");
}

$("#btn-reload").addEventListener("click", () => location.reload());

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    // fetch only rejects when the request never reached a server.
    showOffline("The app is not answering on this address.");
    throw new Error("the app is not running — reload the page");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

let toastTimer = null;

/**
 * How long a message stays up.
 *
 * A fixed four seconds was too short for the ones worth reading — "let go of the
 * 3 files you had picked" is a sentence, not a tick. Long enough to read at a
 * calm pace, with a floor for errors, which arrive when you are already busy
 * being surprised.
 */
function toastMs(message, kind) {
  const reading = 1500 + message.length * 65;   // ~14 characters a second
  return Math.min(Math.max(reading, kind === "error" ? 8000 : 5000), 15000);
}

function toast(message, kind = "") {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = "toast"; }, toastMs(message, kind));
}

// Once it can sit there for ten seconds, it needs a way out from under.
$("#toast").addEventListener("click", () => {
  clearTimeout(toastTimer);
  $("#toast").className = "toast";
});

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function fmtSize(bytes) {
  if (bytes > 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes > 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e3).toFixed(0)} KB`;
}

/* ====================================================================== tabs */

function showView(view) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  if (view === "tune") refreshTunePicker();
  if (view === "translate") loadLibrary();   // may have finished since last look
}

$("#tabs").addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (tab) showView(tab.dataset.view);
});

/* ====================================================================== meta */

// Whisper's languages, common ones first. Detection is unreliable on music, so
// being able to pin matters.
const LANGUAGES = [
  ["", "Detect automatically"],
  ["en", "English"], ["de", "German"], ["es", "Spanish"], ["hi", "Hindi"],
  ["ru", "Russian"], ["ur", "Urdu"], ["pa", "Punjabi"], ["bn", "Bengali"],
  ["ta", "Tamil"], ["te", "Telugu"], ["mr", "Marathi"], ["gu", "Gujarati"],
  ["kn", "Kannada"], ["ml", "Malayalam"], ["ne", "Nepali"], ["si", "Sinhala"],
  ["fr", "French"], ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"],
  ["pl", "Polish"], ["uk", "Ukrainian"], ["tr", "Turkish"], ["ar", "Arabic"],
  ["fa", "Persian"], ["he", "Hebrew"], ["el", "Greek"], ["sv", "Swedish"],
  ["da", "Danish"], ["no", "Norwegian"], ["fi", "Finnish"], ["cs", "Czech"],
  ["sk", "Slovak"], ["hu", "Hungarian"], ["ro", "Romanian"], ["bg", "Bulgarian"],
  ["hr", "Croatian"], ["sr", "Serbian"], ["sl", "Slovenian"], ["zh", "Chinese"],
  ["ja", "Japanese"], ["ko", "Korean"], ["vi", "Vietnamese"], ["th", "Thai"],
  ["id", "Indonesian"], ["ms", "Malay"], ["tl", "Tagalog"], ["sw", "Swahili"],
  ["af", "Afrikaans"], ["ca", "Catalan"], ["lt", "Lithuanian"], ["lv", "Latvian"],
  ["et", "Estonian"], ["is", "Icelandic"], ["hy", "Armenian"], ["ka", "Georgian"],
];

const PROFILE_HINTS = {
  "home-video": "Handheld mics, uneven levels, unclear speech.",
  "music": "Songs, or speech under dense instrumentation. Disables voice activity detection, which otherwise discards nearly all sung audio.",
  "verbatim": "Keeps almost everything. Use when subtitles are missing things you wanted.",
};

const MODEL_HINTS = {
  "large-v3": "Most accurate. The right default.",
  "large-v3-turbo": "3–5× faster, but degrades on exactly the hard audio this is for.",
  "large-v3-int8": "Lower VRAM, some accuracy cost.",
};

async function loadMeta() {
  try {
    state.meta = await api("/api/meta");
  } catch (err) {
    // /api/meta only fills in settings and hints. The file browser needs
    // nothing from it, so failing here must not leave the page unusable —
    // returning early once meant a restarting server killed "Pick files"
    // until a manual reload.
    $("#env").innerHTML =
      `<span class="dot bad"></span> backend error — <a href="#" id="link-retry">retry</a>`;
    document.querySelector("#link-retry")?.addEventListener("click", (event) => {
      event.preventDefault();
      boot();
    });
    toast(`Could not read settings: ${err.message}`, "error");
    return false;
  }

  const { gpu, models, profiles, defaults } = state.meta;
  $("#env").innerHTML = gpu
    ? `<span class="dot ok"></span> ${gpu.name} · ${gpu.vram_gb} GB`
    : `<span class="dot bad"></span> no CUDA GPU`;

  $("#opt-profile").innerHTML = profiles
    .map((p) => `<option value="${p}"${p === defaults.profile ? " selected" : ""}>${p}</option>`)
    .join("");
  updateProfileHint();

  $("#opt-language").innerHTML = LANGUAGES
    .map(([code, name]) =>
      `<option value="${code}"${code === (defaults.language || "") ? " selected" : ""}>${name}</option>`)
    .join("");
  const target = defaults.translate_target || "en";
  $("#opt-translate-target").innerHTML = LANGUAGES
    .filter(([code]) => code)
    .map(([code, name]) =>
      `<option value="${code}"${code === target ? " selected" : ""}>${name}</option>`)
    .join("");

  $("#opt-model").innerHTML = Object.keys(models)
    .map((m) => `<option value="${m}"${m === defaults.model ? " selected" : ""}>${m}</option>`)
    .join("") || `<option value="">no models installed</option>`;
  updateModelHint();

  $("#opt-batch").value = defaults.batch_size;
  $("#opt-beam").value = defaults.beam_size;

  // From settings.local.yaml, so a preference set once is already filled in.
  $("#opt-hotwords").value = defaults.hotwords || "";
  $("#opt-outdir").value = defaults.out_dir || "";
  $("#opt-romanize").checked = !!defaults.romanize;
  $("#opt-keep-suppressed").checked = !!defaults.keep_suppressed;
  const formats = defaults.formats || ["srt", "vtt"];
  $("#fmt-srt").checked = formats.includes("srt");
  $("#fmt-vtt").checked = formats.includes("vtt");

  // If "always translate" was saved, the page opens with it already chosen.
  if (defaults.translate_auto && defaults.translate_provider) {
    $("#opt-translate-mode").value = defaults.translate_provider;
    $("#opt-translate-remember").checked = true;
  }

  buildTuneControls(defaults);
  updateTranslateMode();

  const conf = state.meta.settings;
  if (conf) {
    $("#settings-path").textContent = conf.path;
    if (conf.error) toast(`Settings: ${conf.error}`, "error");
  }
  return true;
}

/** Populate the drive list and open a folder. Independent of everything else. */
async function initBrowser() {
  let drives;
  try {
    drives = await api("/api/drives");
  } catch (err) {
    $("#listing").innerHTML =
      `<li class="muted">Cannot reach the backend — is the app still running?</li>`;
    toast(`Cannot list drives: ${err.message}`, "error");
    return;
  }
  $("#drive-select").innerHTML = [
    `<option value="${drives.home}">Home</option>`,
    ...drives.drives.map((d) => `<option value="${d}">${d}</option>`),
  ].join("");
  await browse(drives.home);
}

function updateProfileHint() {
  $("#profile-hint").textContent = PROFILE_HINTS[$("#opt-profile").value] || "";
}
function updateModelHint() {
  $("#model-hint").textContent = MODEL_HINTS[$("#opt-model").value] || "";
}
/**
 * "Also translate": off, one of the cloud services, or the offline model.
 *
 * The cloud services are the reason this is in Settings at all — they are what
 * you actually want, and their only entry point used to be a panel two screens
 * down. The offline model stays available for footage that must not leave the
 * machine, labelled as the compromise it is.
 */
function translateMode() {
  return $("#opt-translate-mode").value;
}

/**
 * Which target languages an engine can actually reach.
 *
 * Returns null when there is no restriction. Offering a language and then
 * refusing it is worse than not offering it: DeepL has no Urdu, and the only way
 * you found out was an error after choosing it.
 */
function targetsFor(mode) {
  const targets = state.providers?.targets?.[mode];
  return targets && targets.length ? new Set(targets) : null;
}

/** Rebuild a language <select>, disabling what the engine cannot do. */
function fillTargets(select, mode, keep) {
  const allowed = targetsFor(mode);
  const label = mode === "deepl" ? "DeepL" : mode === "local" ? "the offline model" : "";
  select.innerHTML = LANGUAGES
    .filter(([code]) => code)
    .map(([code, name]) => {
      const ok = !allowed || allowed.has(code);
      return `<option value="${code}"${code === keep ? " selected" : ""}`
           + `${ok ? "" : " disabled"}>${name}${ok ? "" : ` — not in ${label}`}</option>`;
    })
    .join("");
  select.value = keep;
  // The saved language may be one this engine cannot reach; fall back to English
  // rather than leaving a disabled option selected and failing later.
  if (allowed && !allowed.has(keep)) {
    select.value = allowed.has("en") ? "en" : [...allowed][0];
  }
}

function updateTranslateMode() {
  const mode = translateMode();
  const cloud = mode === "google" || mode === "deepl";
  if (mode !== "none") {
    fillTargets($("#opt-translate-target"), mode, $("#opt-translate-target").value || "en");
  }
  $("#translate-opts").style.display = mode === "none" ? "none" : "";
  $("#engine-row").style.display = mode === "local" ? "" : "none";
  $("#btn-open-keys").style.display = cloud ? "" : "none";
  $("#remember-row").style.display = mode === "none" ? "none" : "";

  const configured = state.providers?.configured || {};
  const lang = $("#opt-translate-target").value;
  let hint = "";
  if (cloud) {
    const name = mode === "deepl" ? "DeepL" : "Google Translate";
    hint = `Transcribes offline, then sends the subtitle text to ${name}. `
         + "The audio never leaves this machine; the text does.";
    if (!configured[mode]) {
      hint = `${name} has no API key yet — add one first, or the transcription `
           + "will finish untranslated.";
    } else if (mode === "deepl" && state.providers
               && !state.providers.deepl_targets.includes(lang)) {
      hint = `DeepL cannot translate into this language — use Google for it.`;
    }
  } else if (mode === "local") {
    hint = "Runs entirely offline and is clearly weaker than the cloud "
         + "services, especially on slang and idiom. Adds a second pass.";
  }
  if (hint && mode !== "none") {
    const name = LANGUAGES.find(([c]) => c === lang)?.[1] || lang;
    hint += ` Files already in ${name} are left alone.`;
  }
  $("#translate-mode-hint").textContent = hint;
}

/**
 * "Always do this" writes the choice to settings.local.yaml.
 *
 * Kept in that file rather than in the browser: it is where every other default
 * lives, it survives a restart, and it can be read and changed by hand. A
 * setting you cannot find is a setting you cannot turn off.
 */
async function saveTranslateDefault() {
  const on = $("#opt-translate-remember").checked;
  const mode = translateMode();
  try {
    const res = await api("/api/translate/default", {
      method: "POST",
      body: JSON.stringify({
        auto: on && mode !== "none",
        provider: mode === "none" ? "" : mode,
        target: $("#opt-translate-target").value,
      }),
    });
    $("#translate-remember-status").textContent = res.auto
      ? `saved — every non-${res.target} file will be translated with ${res.provider}`
      : "saved — back to asking per run";
  } catch (err) {
    $("#translate-remember-status").textContent = "";
    toast(`Could not save that: ${err.message}`, "error");
    throw err;
  }
}

$("#opt-translate-remember").addEventListener("change", async () => {
  const on = $("#opt-translate-remember").checked;
  try {
    await saveTranslateDefault();
  } catch {
    $("#opt-translate-remember").checked = !on;   // the file did not change
  }
});

/**
 * Keep the saved default in step with the controls while "Always do this" is on.
 *
 * Without this the page could show "Offline model" with the box ticked while the
 * file still said `deepl` — the display claiming one thing and the next run doing
 * another. The tick means "remember what I have selected", so a change to the
 * selection is a change to what is remembered.
 */
async function syncRememberedDefault() {
  if (!$("#opt-translate-remember").checked) return;
  if (translateMode() === "none") {
    $("#opt-translate-remember").checked = false;
  }
  try {
    await saveTranslateDefault();
  } catch { /* already reported */ }
}

/** Jump to where keys are entered, and open the panel so it is not another
 *  click away. This is the button that makes cloud translation findable. */
$("#btn-open-keys").addEventListener("click", () => {
  showView("translate");
  $("#keys-block").open = true;
  const provider = translateMode();
  if (provider === "google" || provider === "deepl") {
    $("#auto-provider").value = provider;
    updateAutoHint();
  }
  $("#keys-block").scrollIntoView({ behavior: "smooth", block: "center" });
  ($("#key-google").offsetParent ? $(`#key-${provider}`) : $("#keys-block")).focus?.();
});
$("#opt-profile").addEventListener("change", updateProfileHint);
$("#opt-model").addEventListener("change", updateModelHint);
$("#opt-translate-mode").addEventListener("change", () => {
  updateTranslateMode();
  syncRememberedDefault();
});
$("#opt-translate-target").addEventListener("change", () => {
  updateTranslateMode();
  syncRememberedDefault();
});

/* =================================================================== browser */

async function browse(path) {
  // Leaving the folder closes the file's tab with it. Checked before the fetch,
  // so cancelling actually leaves you where you were.
  if (path && path !== state.cwd?.path && !leaveFileSettings("open another folder")) return;
  let data;
  try {
    data = await api(`/api/browse?path=${encodeURIComponent(path || "")}`);
  } catch (err) {
    // Put the reason where the click happened. A toast fades after four
    // seconds, which is how a dead backend gets mistaken for a dead button.
    $("#listing").innerHTML =
      `<li class="muted">Cannot open this folder: ${escapeHtml(err.message)}</li>`;
    toast(`Cannot open folder: ${err.message}`, "error");
    return;
  }
  state.cwd = data;
  $("#crumbs").textContent = data.path;
  if (state.scan && state.scan.folder !== data.path) {
    // A scan of the folder we just left says nothing about this one, and neither
    // do the ticks that were made against it.
    state.scan = null;
    state.excluded.clear();
    state.included.clear();
    closeFileSettings();   // it belonged to a file in the folder we just left
    $("#scan-detail").innerHTML = "";
    $("#scan-summary").textContent =
      "Select nothing and the button below takes this whole folder, skipping "
      + "videos that already have subtitles for your current settings.";
  }
  updateSubmitButton();

  const rows = data.dirs.map((dir) =>
    `<li data-dir="${encodeURIComponent(dir.path)}">
       <span class="icon">▸</span><span class="name">${escapeHtml(dir.name)}</span></li>`);

  for (const file of data.files) {
    const selected = state.selection.has(file.path) ? " selected" : "";
    rows.push(`<li class="file${selected}" data-file="${encodeURIComponent(file.path)}"
      data-name="${escapeHtml(file.name)}" data-size="${file.size}">
      <span class="icon">▪</span><span class="name">${escapeHtml(file.name)}</span>
      <span class="size">${fmtSize(file.size)}</span></li>`);
  }
  if (!rows.length) rows.push(`<li class="muted">No folders or media files here.</li>`);
  $("#listing").innerHTML = rows.join("");
}

$("#listing").addEventListener("click", (event) => {
  const li = event.target.closest("li");
  if (!li) return;
  if (li.dataset.dir) {
    browse(decodeURIComponent(li.dataset.dir));
  } else if (li.dataset.file) {
    const path = decodeURIComponent(li.dataset.file);
    const picking = !state.selection.has(path);
    // Picking files by hand and ticking a scan are two ways of saying the same
    // thing, and only one of them can win — with both on screen the ticks were
    // silently ignored. Whichever you did last is the one in charge.
    if (picking && state.scan && !dropScan("the files you pick by hand")) return;
    if (state.selection.has(path)) state.selection.delete(path);
    else state.selection.set(path, { name: li.dataset.name, size: +li.dataset.size });
    li.classList.toggle("selected");
    renderSelection();
  }
});

/** Put the folder scan away, first checking nobody is mid-edit inside it. */
function dropScan(inFavourOf) {
  if (!leaveFileSettings("pick files by hand")) return false;
  const had = state.scan?.files.length || 0;
  state.scan = null;
  state.excluded.clear();
  state.included.clear();
  closeFileSettings();
  $("#scan-detail").innerHTML = "";
  $("#scan-summary").textContent =
    `Checked ${had} file${had === 1 ? "" : "s"} — set aside in favour of `
    + `${inFavourOf}. Press Check this folder again to go back to the whole folder.`;
  return true;
}

$("#btn-up").addEventListener("click", () => {
  if (state.cwd?.parent) browse(state.cwd.parent);
});
$("#drive-select").addEventListener("change", (e) => browse(e.target.value));
// There was a "Select all media" button here. Folder mode replaced it: it took
// this folder only, ignored what was already done, and had no way to reach a
// file's own settings. "Check this folder" then "Tick all" is the same
// instruction, said better — and two buttons that look alike are worse than one.
$("#btn-clear-selection").addEventListener("click", () => {
  state.selection.clear();
  if (state.cwd) browse(state.cwd.path);
  renderSelection();
});
$("#btn-use-current").addEventListener("click", () => {
  if (state.cwd) $("#opt-outdir").value = state.cwd.path;
});

/* ============================================================== folder mode */

/**
 * Options exactly as a submit would send them.
 *
 * The scan has to be judged against the settings that will actually run,
 * otherwise "already done" means something different from what happens next —
 * a file with only Russian subtitles is finished if no translation was asked
 * for, and unfinished if one was.
 */
function currentOptions() {
  const formats = [];
  if ($("#fmt-srt").checked) formats.push("srt");
  if ($("#fmt-vtt").checked) formats.push("vtt");
  return {
    profile: $("#opt-profile").value,
    model: $("#opt-model").value,
    language: $("#opt-language").value,
    hotwords: $("#opt-hotwords").value,
    batch_size: +$("#opt-batch").value,
    beam_size: +$("#opt-beam").value,
    formats,
    romanize: $("#opt-romanize").checked,
    keep_suppressed: $("#opt-keep-suppressed").checked,
    translate: translateMode() === "local",
    cloud_provider:
      translateMode() === "google" || translateMode() === "deepl"
        ? translateMode() : "",
    translate_engine: $("#opt-translate-engine").value,
    translate_target: $("#opt-translate-target").value,
    overwrite: $("#opt-overwrite").checked,
  };
}

const STATE_LABELS = {
  done: "already done",
  pending: "to transcribe",
  translate: "translation only",
  damaged: "interrupted — will be redone",
};

/**
 * Re-check the folder.
 *
 * `quiet` is for a re-scan caused by editing one file's settings: it leaves the
 * list standing and patches the rows in place. Rebuilding it would collapse the
 * page to nothing for a moment, which on a folder of fifty videos means being
 * thrown back to the top and scrolling down again after every single choice.
 */
async function scanFolder({ quiet = false } = {}) {
  if (!state.cwd) return null;
  if (!quiet) {
    // The other way of choosing. Leaving hand-picked files in place would mean
    // the tick boxes below them decided nothing.
    if (state.selection.size) {
      const had = state.selection.size;
      state.selection.clear();
      renderSelection();
      if (state.cwd) await browse(state.cwd.path);
      toast(`Let go of the ${had} file${had === 1 ? "" : "s"} you had picked — `
            + "the tick boxes decide now.", "");
    }
    $("#scan-summary").textContent = "checking…";
    $("#scan-detail").innerHTML = "";
  }
  try {
    const scan = await api("/api/scan", {
      method: "POST",
      body: JSON.stringify({
        folder: state.cwd.path,
        options: currentOptions(),
        out_dir: $("#opt-outdir").value.trim() || null,
        recursive: $("#opt-recursive").checked,
      }),
    });
    state.scan = scan;
    $("#scan-summary").textContent = scan.summary;

    // The list first: the button's label counts the boxes in it.
    if (!(quiet && patchScanList(scan))) renderScanList(scan);
    if (state.fileSettings) renderFileSettings();
    updateSubmitButton();
    return scan;
  } catch (err) {
    $("#scan-summary").textContent = `Could not check the folder: ${err.message}`;
    state.scan = null;
    updateSubmitButton();
    return null;
  }
}

/**
 * The scan list, with a checkbox per video.
 *
 * Everything the scan found is listed, not only the unfinished ones: the useful
 * question is often "skip that one" or "redo that one", and neither is
 * answerable from a read-only summary. Files needing work start ticked, finished
 * files start unticked — so pressing the button does the obvious thing without
 * any clicking, and every deviation from it is one click.
 */
function renderScanList(scan) {
  if (!scan.files.length) {
    $("#scan-detail").innerHTML = "";
    return;
  }
  // The list scrolls inside its own box, so rebuilding it lands you back at the
  // top of fifty videos. Anything still on screen afterwards keeps its place.
  const keep = scanViewState();
  const problems = (scan.config_problems || []).length
    ? `<p class="hint scan-problem">${scan.config_problems
        .map((p) => escapeHtml(p)).join("<br>")}</p>`
    : "";
  $("#scan-detail").innerHTML = `
    <div class="scan-actions">
      <button class="btn ghost tiny" id="btn-scan-all">Tick all</button>
      <button class="btn ghost tiny" id="btn-scan-none">Untick all</button>
      <button class="btn ghost tiny" id="btn-scan-todo">Only the unfinished</button>
      ${resetButtonHtml(scan)}
    </div>
    ${problems}
    <ul class="scan-list">${scan.files.map((f) => scanRowHtml(f)).join("")}</ul>
    <p class="hint">
      <strong>Settings…</strong> opens that file on its own tab in the settings
      panel. Anything you change there is saved in
      <code>${escapeHtml(scan.config_file || "sgen.folder.yaml")}</code> beside the
      videos, so it survives restarts, moves with the folder, and can be edited by
      hand for a large batch.
    </p>`;
  restoreScanView(keep);
}

/** Which rows are expanded, and how far down the list you were. */
function scanViewState() {
  return {
    scroll: find("#scan-detail .scan-list")?.scrollTop || 0,
    open: $$("#scan-detail li[data-path] details[open]")
      .map((d) => d.closest("li").dataset.path),
  };
}

function restoreScanView(keep) {
  for (const path of keep.open) {
    const details = find(`#scan-detail li[data-path="${path}"] details`);
    if (details) details.open = true;
  }
  const list = find("#scan-detail .scan-list");
  if (list) list.scrollTop = keep.scroll;
}

/**
 * A row in the scan list.
 *
 * The row says what will happen to the file and offers a way in. It does not
 * try to be a settings form: three controls squeezed into a list row were
 * cramped, could only reach half the settings, and put the same choice in two
 * different-looking places. The button opens the real thing.
 */
function scanRowHtml(f) {
  const path = encodeURIComponent(f.path);
  const own = f.overrides || {};
  const names = ownNames(own);
  const open = state.fileSettings === f.path ? " scan-open" : "";
  return `<li class="scan-${f.state}${names ? " scan-custom" : ""}${open}" data-path="${path}">
      <label class="check scan-pick">
        <input type="checkbox" data-scan-path="${path}"
          ${scanRowTicked(f) ? "checked" : ""}>
        <span class="scan-name">${escapeHtml(f.name)}</span>
      </label>
      <span class="scan-state">${STATE_LABELS[f.state] || f.state}</span>
      <span class="scan-why">${escapeHtml(f.reason)}</span>
      <span class="scan-own-line">
        <button class="btn ghost tiny" data-open-settings="${path}">Settings…</button>
        <span class="scan-own-note">${names ? `its own: ${escapeHtml(names)}` : ""}</span>
      </span>
    </li>`;
}

/** Ticked by default when there is work to do, minus the user's own deviations. */
function scanRowTicked(f) {
  return f.state === "done"
    ? state.included.has(f.path)
    : !state.excluded.has(f.path);
}

// Naming the settings beats counting them: "its own: profile, translate" says
// what is different without opening anything.
const SETTING_NAMES = {
  profile: "profile",
  language: "language",
  hotwords: "names",
  romanize: "Latin script",
  translate: "translation",
  translate_target: "target language",
};

// Listed in the order they appear on the tab, not the order the YAML happens to
// store them in.
const ownNames = (own) =>
  Object.keys(SETTING_NAMES)
    .filter((key) => key in (own || {}))
    .map((key) => SETTING_NAMES[key])
    .join(", ");

// Only offered when there is something to undo, so the button is never a decoy —
// and it names how many files, because it discards work you did. "3 files" and
// not "(3)": the same bare number one line down counts settings, not files.
function resetButtonHtml(scan) {
  const custom = scan.files.filter((f) => Object.keys(f.overrides || {}).length).length;
  return custom
    ? `<button class="btn ghost tiny" id="btn-reset-overrides">
         Reset ${custom} file${custom === 1 ? "" : "s"} to All files</button>`
    : "";
}

/**
 * Bring an existing list up to date without rebuilding it.
 *
 * Only the parts that a per-file change can affect are touched — the state
 * label, the reason, the summary count, the tick — so nothing moves, no
 * `<details>` snaps shut, and the control just used keeps focus. Returns false
 * when the list on screen no longer matches the scan (a file appeared or
 * vanished on disk), which needs a real render.
 */
function patchScanList(scan) {
  const rows = new Map($$("#scan-detail li[data-path]").map((li) => [li.dataset.path, li]));
  if (rows.size !== scan.files.length) return false;
  if (!scan.files.every((f) => rows.has(encodeURIComponent(f.path)))) return false;
  // A complaint about the config file appearing or clearing changes the layout.
  const hasProblems = (scan.config_problems || []).length > 0;
  if (hasProblems !== !!find("#scan-detail .scan-problem")) return false;
  const actions = find("#scan-detail .scan-actions");
  if (!actions) return false;

  for (const f of scan.files) patchScanRow(rows.get(encodeURIComponent(f.path)), f);

  const reset = find("#btn-reset-overrides");
  const wanted = resetButtonHtml(scan);
  const label = wanted.replace(/<[^>]*>/g, "").trim();
  if (!wanted) {
    reset?.remove();
  } else if (!reset) {
    actions.insertAdjacentHTML("beforeend", wanted);
  } else if (reset.dataset.confirming !== "yes") {
    reset.textContent = label;
  } else if (reset.dataset.label !== label) {
    // It is armed and the count moved under it. Disarm rather than relabel: a
    // stale count on an armed button resets more files than it offered to.
    disarm(reset);
    reset.textContent = label;
  }
  return true;
}

function patchScanRow(li, f) {
  const own = f.overrides || {};
  const names = ownNames(own);
  const open = state.fileSettings === f.path ? " scan-open" : "";

  li.className = `scan-${f.state}${names ? " scan-custom" : ""}${open}`;
  li.querySelector(".scan-state").textContent = STATE_LABELS[f.state] || f.state;
  li.querySelector(".scan-why").textContent = f.reason;
  // Naming what is different is the row's whole job now the controls have moved.
  li.querySelector(".scan-own-note").textContent = names ? `its own: ${names}` : "";
  set(li.querySelector("input[data-scan-path]"), "checked", scanRowTicked(f));
}

/** Assign only on a real change: writing to a live control can disturb it. */
function set(el, prop, value) {
  if (el && el[prop] !== value) el[prop] = value;
}

/** Write one file's overrides to the folder's config file, then re-scan. */
async function setOverride(path, values) {
  try {
    await api("/api/folder-config", {
      method: "POST",
      body: JSON.stringify({ folder: state.cwd.path, path, values }),
    });
    // Re-scan, because a per-file change can flip whether that file counts as
    // finished — but quietly, patching the rows where they are. A rebuild here
    // scrolled the page back to the top after every choice.
    await scanFolder({ quiet: true });
    return true;
  } catch (err) {
    toast(`Could not save that file's settings: ${err.message}`, "error");
    return false;
  }
}

/* ======================================================= one file's settings */

/**
 * Per-file settings live in the settings panel, on their own tab.
 *
 * They used to be three controls inside a row of the scan list. That was cramped
 * enough to reach only half of what can vary per file, and it put the same
 * decision in two places that looked nothing like each other. Here they are the
 * same controls in the same panel as everything else, each one able to say "as
 * in All files" — which is what an override actually is.
 */
$("#scan-detail").addEventListener("click", (event) => {
  const open = event.target.closest("[data-open-settings]");
  if (!open) return;
  const path = decodeURIComponent(open.dataset.openSettings);
  // Opening another file abandons this one's edits, so ask first.
  if (path !== state.fileSettings && !leaveFileSettings("open another file")) return;
  openFileSettings(path);
});

$("#tab-global").addEventListener("click", () => {
  if (leaveFileSettings("switch tabs")) showSettingsTab("global");
});
$("#tab-file").addEventListener("click", () => showSettingsTab("file"));

// "as in All files (music)" has to keep telling the truth when All files changes.
$("#settings-global").addEventListener("change", () => {
  if (state.fileSettings) renderFileSettings();
});

function showSettingsTab(which) {
  const onFile = which === "file" && !!state.fileSettings;
  $("#settings-global").classList.toggle("hidden", onFile);
  $("#settings-file").classList.toggle("hidden", !onFile);
  $("#tab-global").classList.toggle("active", !onFile);
  $("#tab-file").classList.toggle("active", onFile);
}

function openFileSettings(path) {
  const file = state.scan?.files.find((f) => f.path === path);
  if (!file) return;
  state.fileSettings = path;
  state.fileDirty = false;
  renderFileSettings();
  showSettingsTab("file");
  // Mark the row it belongs to, so the two panels are visibly connected.
  if (state.scan) patchScanList(state.scan);
  find("#settings-file")?.scrollIntoView({ block: "nearest" });
}

function closeFileSettings() {
  state.fileSettings = null;
  state.fileDirty = false;
  $("#tab-file").classList.add("hidden");
  showSettingsTab("global");
}

/* --------------------------------------------------------- unsaved changes

   Settings used to be written the moment a dropdown moved, which made a misclick
   permanent and silent. Now the tab is a form: nothing reaches the folder file
   until Save, and every way out of the tab asks first. Cancel keeps you here —
   the safe answer is the one that does nothing.                              */

/** True to proceed. Asks only when there is something to lose. */
function leaveFileSettings(what) {
  if (!state.fileDirty) return true;
  const name = openFile()?.name || "this file";
  const ok = window.confirm(
    `${name} has unsaved settings.\n\n`
    + `OK discards them and continues to ${what}. Cancel stays here.`,
  );
  if (ok) {
    state.fileDirty = false;
    renderFileSettings();
  }
  return ok;
}

// Closing the tab or reloading is the one case the page cannot handle itself.
window.addEventListener("beforeunload", (event) => {
  if (!state.fileDirty) return;
  event.preventDefault();
  event.returnValue = "";
});

/** The file currently open on the second tab, as the scan last reported it. */
function openFile() {
  return state.scan?.files.find((f) => f.path === state.fileSettings) || null;
}

const langName = (code) =>
  LANGUAGES.find(([c]) => c === code)?.[1] || code || "detect automatically";

const TRANSLATE_LABELS = {
  none: "no translation", deepl: "DeepL", google: "Google",
  local: "offline model",
};

/**
 * Fill the tab from the file's saved settings, with every control offering the
 * inherited value by name — "as in All files (music)" says what will happen if
 * you leave it alone, which a blank option does not.
 */
function renderFileSettings() {
  const file = openFile();
  if (!file) return closeFileSettings();
  const saved = file.overrides || {};
  // While there are unsaved edits, the form shows those — a re-scan or a change
  // on the other tab must not quietly undo what you were in the middle of.
  const own = state.fileDirty ? fileSettingValues() : saved;
  const inherit = (label) => `as in All files (${label})`;

  $("#file-settings-name").textContent = file.name;
  $("#tab-file-name").textContent = file.name;
  $("#tab-file").title = file.path;
  $("#tab-file").classList.remove("hidden");

  fillSelect($("#f-profile"), [
    ["", inherit($("#opt-profile").value)],
    ...(state.meta?.profiles || []).map((p) => [p, p]),
  ], own.profile || "");

  // "auto" is storable, so a file can be detected even when All files pins a
  // language. An empty value cannot be stored, so it has to mean "inherit".
  fillSelect($("#f-language"), [
    ["", inherit(langName($("#opt-language").value))],
    ["auto", "Detect automatically"],
    ...LANGUAGES.filter(([code]) => code),
  ], own.language || "");

  // Assign only on a change: this re-renders while you are still typing.
  set($("#f-hotwords"), "value", own.hotwords || "");
  set($("#f-hotwords"), "placeholder",
      `as in All files (${$("#opt-hotwords").value.trim() || "none"})`);

  fillSelect($("#f-romanize"), [
    ["", inherit($("#opt-romanize").checked ? "yes" : "no")],
    ["yes", "Yes"],
    ["no", "No"],
  ], own.romanize === undefined ? "" : (own.romanize ? "yes" : "no"));

  fillSelect($("#f-translate"), [
    ["", inherit(TRANSLATE_LABELS[translateMode()] || translateMode())],
    ...Object.entries(TRANSLATE_LABELS),
  ], own.translate || "");

  // Same engine-aware list as the other tab: a language the engine cannot reach
  // is shown disabled and says why, rather than failing after you pick it.
  const mode = own.translate || translateMode();
  const target = $("#f-translate-target");
  // Filled even while hidden: an empty <select> is one race away from being
  // shown empty when the engine changes.
  fillTargets(target, mode, own.translate_target || $("#opt-translate-target").value || "en");
  target.insertAdjacentHTML("afterbegin",
    `<option value="">${escapeHtml(inherit(langName($("#opt-translate-target").value)))}</option>`);
  target.value = own.translate_target || "";

  $("#f-translate-opts").style.display = mode === "none" ? "none" : "";
  $("#f-profile-hint").textContent = PROFILE_HINTS[own.profile || $("#opt-profile").value] || "";
  $("#f-translate-hint").textContent = own.translate === "none"
    ? "This file is left in its own language, whatever All files says."
    : "";
  $("#file-settings-path").textContent = state.scan?.config_file || "sgen.folder.yaml";
  updateFileSaveState();
}

/** The Save button, the discard button, the tab badge and the status line. */
function updateFileSaveState() {
  const saved = openFile()?.overrides || {};
  const dirty = state.fileDirty;
  const savedNames = ownNames(saved);

  $("#btn-file-save").disabled = !dirty;
  $("#btn-file-save").textContent = dirty ? "Save" : "Saved";
  $("#btn-file-revert").classList.toggle("hidden", !dirty);
  $("#btn-file-clear").classList.toggle("hidden", !savedNames && !dirty);

  const badge = $("#tab-file-count");
  badge.textContent = dirty ? "unsaved" : (Object.keys(saved).length || "");
  badge.classList.toggle("unsaved", dirty);
  badge.classList.toggle("hidden", !dirty && !savedNames);

  $("#file-settings-status").textContent = dirty
    ? "Not saved yet — Save applies these, Discard puts them back."
    : (savedNames ? `Saved: ${savedNames}.` : "Nothing set here — this file follows All files.");
  $("#file-settings-status").classList.toggle("warn-text", dirty);
}

/** Two sets of overrides are the same when they say the same thing. */
function sameSettings(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].every((key) => a[key] === b[key]);
}

function fillSelect(select, options, selected) {
  select.innerHTML = options
    .map(([value, label]) =>
      `<option value="${value}"${value === selected ? " selected" : ""}>${escapeHtml(label)}</option>`)
    .join("");
  select.value = selected;
}

/** Everything the tab has set, as the folder config stores it. */
function fileSettingValues() {
  const values = {};
  const text = (id) => $(id).value.trim();
  if (text("#f-profile")) values.profile = text("#f-profile");
  if (text("#f-language")) values.language = text("#f-language");
  if (text("#f-hotwords")) values.hotwords = text("#f-hotwords");
  if (text("#f-romanize")) values.romanize = text("#f-romanize") === "yes";
  if (text("#f-translate")) values.translate = text("#f-translate");
  if (text("#f-translate-target")) values.translate_target = text("#f-translate-target");
  return values;
}

/** Any edit marks the form dirty; nothing is written until Save. */
function noteFileEdit() {
  if (!state.fileSettings) return;
  const saved = openFile()?.overrides || {};
  const dirty = !sameSettings(fileSettingValues(), saved);
  if (dirty === state.fileDirty) return;
  state.fileDirty = dirty;
  renderFileSettings();
}

$("#settings-file").addEventListener("change", (event) => {
  if (event.target.closest("[data-file-setting]")) noteFileEdit();
});
$("#f-hotwords").addEventListener("input", noteFileEdit);
$("#f-hotwords").addEventListener("keydown", (event) => {
  if (event.key === "Enter") saveFileSettings();
});

// Ctrl+S is what hands reach for in a form that has a Save button.
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s"
      && state.fileSettings && state.fileDirty) {
    event.preventDefault();
    saveFileSettings();
  }
});

$("#btn-file-save").addEventListener("click", () => saveFileSettings());

async function saveFileSettings() {
  const path = state.fileSettings;
  if (!path || !state.fileDirty) return;
  const values = fileSettingValues();
  $("#btn-file-save").disabled = true;
  $("#btn-file-save").textContent = "Saving…";
  if (await setOverride(path, values)) {
    state.fileDirty = false;
    toast(Object.keys(values).length
      ? `Saved: ${ownNames(values)}.`
      : "Saved — this file follows All files.", "ok");
  }
  renderFileSettings();
}

$("#btn-file-revert").addEventListener("click", () => {
  state.fileDirty = false;
  renderFileSettings();
  toast("Changes discarded.", "");
});

// Puts every control back to "as in All files" — as an edit, not as a write.
// Everything on this tab now goes through Save, including undoing everything.
$("#btn-file-clear").addEventListener("click", () => {
  if (!state.fileSettings) return;
  for (const control of $$("#settings-file [data-file-setting]")) control.value = "";
  noteFileEdit();
  if (!state.fileDirty) toast("Already following All files.", "");
});

/**
 * Reset every file in the folder to the panel's settings.
 *
 * Two clicks, because it throws away choices that took effort to make. The
 * button states the count both times, so "reset 12" is never mistaken for
 * "reset the one I am looking at" — and the tab's own "Use the All files
 * settings" is still there when only one file went wrong.
 */
$("#scan-detail").addEventListener("click", async (event) => {
  const btn = event.target.closest("#btn-reset-overrides");
  if (!btn) return;

  const count = btn.textContent.match(/\d+/)?.[0] || "";
  if (!arm(btn, `Really reset all ${count}?`)) return;

  btn.disabled = true;
  try {
    const res = await api("/api/folder-config/reset", {
      method: "POST",
      body: JSON.stringify({ folder: state.cwd.path }),
    });
    toast(res.cleared
      ? `${res.cleared} file${res.cleared === 1 ? "" : "s"} back on the All files settings.`
      : "Nothing to reset.", "ok");
    await scanFolder({ quiet: true });
  } catch (err) {
    toast(`Could not reset: ${err.message}`, "error");
    btn.disabled = false;
    disarm(btn);
  }
});

/** Paths the user has ticked — what a folder submit will actually queue. */
function scanChecked() {
  return $$("#scan-detail input[data-scan-path]:checked")
    .map((box) => decodeURIComponent(box.dataset.scanPath));
}

$("#scan-detail").addEventListener("change", (event) => {
  const box = event.target.closest("input[data-scan-path]");
  if (!box) return;
  const path = decodeURIComponent(box.dataset.scanPath);
  const file = state.scan?.files.find((f) => f.path === path);
  // Remember the deviation, not the state, so a re-scan keeps the user's intent.
  if (file?.state === "done") {
    box.checked ? state.included.add(path) : state.included.delete(path);
  } else {
    box.checked ? state.excluded.delete(path) : state.excluded.add(path);
  }
  updateSubmitButton();
});

$("#scan-detail").addEventListener("click", (event) => {
  const id = event.target.id;
  if (!["btn-scan-all", "btn-scan-none", "btn-scan-todo"].includes(id)) return;
  for (const box of $$("#scan-detail input[data-scan-path]")) {
    const path = decodeURIComponent(box.dataset.scanPath);
    const file = state.scan?.files.find((f) => f.path === path);
    const wanted = id === "btn-scan-all" ? true
                 : id === "btn-scan-none" ? false
                 : file?.state !== "done";
    box.checked = wanted;
    if (file?.state === "done") {
      wanted ? state.included.add(path) : state.included.delete(path);
    } else {
      wanted ? state.excluded.delete(path) : state.excluded.add(path);
    }
  }
  updateSubmitButton();
});

$("#btn-scan-folder").addEventListener("click", () => scanFolder());
$("#opt-recursive").addEventListener("change", () => {
  if (state.scan) scanFolder();
  updateSubmitButton();
});

function renderSelection() {
  const count = state.selection.size;
  $("#selection-count").textContent = count;
  $("#selection-list").innerHTML = Array.from(state.selection.entries())
    .map(([path, meta]) =>
      `<li>${escapeHtml(meta.name)}<button data-drop="${encodeURIComponent(path)}">×</button></li>`)
    .join("");
  updateSubmitButton();
}

/**
 * One button, two meanings — because two buttons that both said "Transcribe N
 * files" was worse.
 *
 * A selection and a scan can no longer both exist — starting one lets go of the
 * other — so this reads whichever is in charge. With neither, the button takes
 * the whole folder and skips what is already done, which is the common case for
 * a directory of holiday videos.
 */
function updateSubmitButton() {
  const btn = $("#btn-submit");
  const count = state.selection.size;
  if (count) {
    btn.disabled = false;
    btn.textContent = `Transcribe ${count} selected file${count === 1 ? "" : "s"}`;
    return;
  }
  if (!state.cwd) {
    btn.disabled = true;
    btn.textContent = "Transcribe";
    return;
  }
  // A scan is only a preview; the button works without one, and the server
  // decides again at submit time so the two can never disagree.
  const scan = state.scan && state.scan.folder === state.cwd.path ? state.scan : null;
  if (scan) {
    const ticked = scanChecked().length;
    btn.disabled = ticked === 0;
    btn.textContent = ticked
      ? `Transcribe ${ticked} file${ticked === 1 ? "" : "s"}`
      : (scan.total === scan.counts.done ? "This folder is done" : "Nothing ticked");
    return;
  }
  btn.disabled = false;
  btn.textContent = "Transcribe this folder";
}

$("#selection-list").addEventListener("click", (event) => {
  const drop = event.target.dataset.drop;
  if (!drop) return;
  state.selection.delete(decodeURIComponent(drop));
  if (state.cwd) browse(state.cwd.path);
  renderSelection();
});

/* ==================================================================== submit */

$("#btn-submit").addEventListener("click", async () => {
  const options = currentOptions();
  if (!options.formats.length) return toast("Pick at least one output format.", "error");
  // Unsaved per-file settings would not be used by the run — the worst moment to
  // find that out is after the GPU has finished.
  if (!leaveFileSettings("start transcribing")) return;

  const folderMode = state.selection.size === 0;
  if (folderMode && !state.cwd) return;
  // With a scan on screen, the ticked boxes are the instruction — send those
  // paths rather than the folder, so unticking one actually excludes it.
  const scan = state.scan && state.scan.folder === state.cwd?.path ? state.scan : null;
  const ticked = folderMode && scan ? scanChecked() : null;
  if (ticked && !ticked.length) return;

  const btn = $("#btn-submit");
  btn.disabled = true;
  try {
    const res = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        paths: ticked
          ? ticked
          : folderMode ? [state.cwd.path] : Array.from(state.selection.keys()),
        out_dir: $("#opt-outdir").value.trim() || null,
        options,
        // Only an unreviewed whole folder gets the skip. Once files are ticked
        // by hand — in the list or the scan — they run because they were picked.
        skip_done: folderMode && !ticked,
        recursive: $("#opt-recursive").checked,
      }),
    });
    const skipped = res.skipped_count
      ? `, skipped ${res.skipped_count} already done` : "";
    toast(
      res.jobs.length
        ? `Queued ${res.jobs.length} file${res.jobs.length === 1 ? "" : "s"}${skipped}.`
        : `Nothing to do — all ${res.skipped_count} files already have subtitles.`,
      res.jobs.length ? "ok" : "",
    );
    state.selection.clear();
    if (state.cwd) await browse(state.cwd.path);
    if (folderMode && state.scan) await scanFolder();
    renderSelection();
  } catch (err) {
    toast(`Could not queue: ${err.message}`, "error");
  } finally {
    updateSubmitButton();
  }
});

$("#btn-clear-jobs").addEventListener("click", async () => {
  await api("/api/jobs/clear", { method: "POST" });
  for (const [id, job] of state.jobs) {
    if (["done", "failed", "cancelled"].includes(job.status)) state.jobs.delete(id);
  }
  renderJobs();
});

/* ==================================================================== events */

function connectEvents() {
  const source = new EventSource("/api/events");

  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "hello") {
      state.jobs.clear();
      for (const job of payload.jobs) state.jobs.set(job.id, job);
      renderJobs();
      refreshTunePicker();
    } else if (payload.type === "job") {
      const previous = state.jobs.get(payload.job.id);
      state.jobs.set(payload.job.id, payload.job);
      renderJobs();
      if (previous?.status !== "done" && payload.job.status === "done") {
        refreshTunePicker();
        loadLibrary();   // a new transcript is now on disk
        toast(payload.job.suspect
          ? `${payload.job.name}: result looks wrong — see the warning`
          : `${payload.job.name}: ${payload.job.cue_count} subtitles`,
          payload.job.suspect ? "error" : "ok");
      }
      if (previous?.status !== "failed" && payload.job.status === "failed") {
        toast(`${payload.job.name} failed: ${payload.job.error}`, "error");
      }
    }
  };

  // EventSource retries forever and never gives up on its own, so a page whose
  // server has gone sits on "reconnecting…" indefinitely. Two failures in a row
  // is enough to say so plainly.
  let failures = 0;
  source.onerror = () => {
    failures += 1;
    $("#env").innerHTML = `<span class="dot pending"></span> reconnecting…`;
    if (failures >= 2) showOffline("The connection to the app dropped.");
  };
  source.onopen = () => {
    failures = 0;
    hideOffline();
    if (state.meta?.gpu) {
      $("#env").innerHTML =
        `<span class="dot ok"></span> ${state.meta.gpu.name} · ${state.meta.gpu.vram_gb} GB`;
    }
  };
}

const STAGE_LABELS = {
  probe: "Reading file", extract: "Extracting audio", detect: "Detecting language",
  transcribe: "Transcribing", gate: "Checking", translate: "Translating",
  write: "Writing subtitles", done: "Done",
};

function renderJobs() {
  const jobs = Array.from(state.jobs.values());
  if (!jobs.length) {
    $("#jobs").innerHTML = "";
    return;
  }

  $("#jobs").innerHTML = jobs.map((job) => {
    const pct = Math.round((job.progress || 0) * 100);
    const parts = [];
    if (job.status === "running") {
      const stage = STAGE_LABELS[job.stage] || job.stage || "";
      parts.push(stage + (job.stage === "transcribe"
        ? ` ${Math.round(job.stage_fraction * 100)}%` : ""));
    }
    if (job.language) parts.push(`${job.language} ${Math.round(job.language_probability * 100)}%`);
    if (job.cue_count) parts.push(`${job.cue_count} subtitles`);
    if (job.status === "done") parts.push(`${Math.round((job.coverage || 0) * 100)}% covered`);
    if (job.suppressed_count) parts.push(`${job.suppressed_count} gated`);
    if (job.speed) parts.push(`${job.speed}× realtime`);

    const translateBtn = job.status === "done" && job.content_id
      ? `<button class="btn tiny" data-translate="${job.content_id}"
           data-name="${escapeHtml(job.name)}">Translate…</button>` : "";
    const cancel = job.status === "queued"
      ? `<button class="btn ghost tiny" data-cancel="${job.id}">Cancel</button>` : "";

    const warn = job.suspect
      ? `<div class="job-warn"><strong>⚠ This result looks wrong</strong>
           ${(job.qc_notes || []).map((n) => `<div>${escapeHtml(n)}</div>`).join("")}</div>`
      : "";
    const outputs = (job.outputs || []).length
      ? `<div class="job-outputs">${job.outputs.map((o) =>
          `<div>→ ${escapeHtml(o)}</div>`).join("")}</div>` : "";
    const error = job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : "";
    // Say what the cloud pass did. A translation that quietly did not happen is
    // worse than one that failed loudly.
    const benign = /^(translated|already in)/.test(job.cloud_note || "");
    const cloud = job.cloud_note
      ? `<div class="${benign ? "job-detail" : "job-error"}">
           ${escapeHtml(job.cloud_note)}</div>`
      : "";
    const bar = job.status === "running"
      ? `<div class="bar"><i style="width:${pct}%"></i></div>` : "";

    return `<div class="job${job.suspect ? " suspect" : ""}">
      <div class="job-head">
        <span class="job-name" title="${escapeHtml(job.path)}">${escapeHtml(job.name)}</span>
        <span class="job-status ${job.suspect ? "suspect" : job.status}">${job.suspect ? "check this" : job.status}</span>
        ${translateBtn}${cancel}
      </div>
      ${bar}
      ${parts.length ? `<div class="job-detail">${parts.map((p) => `<span>${escapeHtml(p)}</span>`).join("")}</div>` : ""}
      ${warn}${error}${cloud}${outputs}
    </div>`;
  }).join("");
}

$("#jobs").addEventListener("click", async (event) => {
  const contentId = event.target.dataset.translate;
  if (contentId) {
    openTranslate(contentId, event.target.dataset.name || "");
    return;
  }
  const id = event.target.dataset.cancel;
  if (!id) return;
  try {
    await api(`/api/jobs/${id}`, { method: "DELETE" });
  } catch (err) {
    toast(err.message, "error");
  }
});

/* =================================================================== library */

/**
 * Transcripts already on disk.
 *
 * The queue is in-memory, so a server restart forgot every finished file — and
 * the Translate button only exists on a finished file. Reading the sidecars
 * keeps everything reachable across restarts.
 */
async function loadLibrary() {
  let items;
  try {
    items = (await api("/api/library")).items;
  } catch (err) {
    $("#library").innerHTML =
      `<p class="muted">Could not read past transcripts: ${escapeHtml(err.message)}</p>`;
    return;
  }
  state.library = items;
  renderLibraryNote(items);

  // Anything already showing as a live job doesn't need repeating here.
  const live = new Set(Array.from(state.jobs.values()).map((j) => j.content_id).filter(Boolean));
  const rest = items.filter((i) => !live.has(i.content_id));

  renderTranslatePicker(items);

  $("#library-heading").style.display = rest.length ? "" : "none";
  if (!rest.length) {
    $("#library").innerHTML = `<p class="muted">Nothing transcribed yet.</p>`;
    return;
  }

  $("#library").innerHTML = rest.map((item) => {
    const mins = item.duration ? `${Math.round(item.duration / 60)} min` : "";
    const missing = item.source_exists ? "" :
      `<span class="job-status cancelled" title="${escapeHtml(item.path)}">source moved</span>`;
    return `<div class="job">
      <div class="job-head">
        <span class="job-name" title="${escapeHtml(item.path)}">${escapeHtml(item.name)}</span>
        ${missing}
        <button class="btn tiny" data-translate="${item.content_id}"
          data-name="${escapeHtml(item.name)}">Translate…</button>
        ${forgetButton(item)}
      </div>
      <div class="job-detail">
        <span>${escapeHtml(item.language || "?")}</span>
        <span>${item.cue_count} subtitles</span>
        ${mins ? `<span>${mins}</span>` : ""}
        ${item.size ? `<span>${fmtSize(item.size)} cached</span>` : ""}
      </div>
    </div>`;
  }).join("");
}

$("#library").addEventListener("click", libraryClick);

/**
 * The same list, on the Translate tab.
 *
 * The Translate buttons used to exist only at the bottom of the third panel,
 * about 1300px down — far enough that the feature read as missing. Every file
 * is listed here too, one click from the top of the page.
 */
function renderTranslatePicker(items) {
  const target = $("#translate-picker");
  if (!items.length) {
    target.innerHTML =
      `<p class="muted">Nothing transcribed yet. Transcribe a file on the
       <strong>Subtitles</strong> tab first — translation works from the stored
       transcript, so it needs no GPU and no second pass.</p>`;
    return;
  }
  target.innerHTML = items.map((item) => `
    <div class="job">
      <div class="job-head">
        <span class="job-name" title="${escapeHtml(item.path)}">${escapeHtml(item.name)}</span>
        <button class="btn tiny primary" data-translate="${item.content_id}"
          data-name="${escapeHtml(item.name)}">Translate…</button>
        ${forgetButton(item)}
      </div>
      <div class="job-detail">
        <span>${escapeHtml(item.language || "?")}</span>
        <span>${item.cue_count} subtitles</span>
        ${item.duration ? `<span>${Math.round(item.duration / 60)} min</span>` : ""}
        ${item.size ? `<span>${fmtSize(item.size)} cached</span>` : ""}
      </div>
    </div>`).join("");
}

$("#translate-picker").addEventListener("click", libraryClick);

/* ------------------------------------------------------------------ forgetting

   The list is a record of private footage: the sidecar holds the full text of
   what was said and the path it came from, and the WAV holds the audio. Deleting
   it needs to be a button here, not an explanation of which folder to open.     */

function forgetButton(item) {
  return `<button class="btn ghost tiny" data-forget="${item.content_id}"
    data-name="${escapeHtml(item.name)}"
    title="Delete the cached transcript and audio for this file">Forget</button>`;
}

/** Both lists share this: Translate… on one button, Forget on the other. */
function libraryClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.translate) {
    openTranslate(button.dataset.translate, button.dataset.name || "");
  } else if (button.dataset.forget) {
    forgetFile(button);
  }
}

/**
 * Two clicks, because it destroys a transcript that took GPU minutes to make
 * and cannot be rebuilt without the source file — which may itself be gone.
 */
async function forgetFile(button) {
  if (!arm(button, "Delete transcript?")) return;
  button.disabled = true;
  try {
    const res = await api(`/api/library/${encodeURIComponent(button.dataset.forget)}`,
                          { method: "DELETE" });
    toast(`Forgot ${res.name}${res.freed ? ` — ${fmtSize(res.freed)} freed` : ""}.`, "ok");
    await loadLibrary();
  } catch (err) {
    toast(`Could not forget it: ${err.message}`, "error");
    button.disabled = false;
    disarm(button);
  }
}

$("#btn-forget-all").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const count = state.library?.length || 0;
  if (!arm(button, `Delete all ${count} transcripts?`)) return;
  button.disabled = true;
  try {
    const res = await api("/api/library/forget-all", {
      method: "POST",
      body: JSON.stringify({ confirm: true }),
    });
    const freed = res.freed ? ` — ${fmtSize(res.freed)} freed` : "";
    toast(res.removed
      ? `Forgot ${res.removed} file${res.removed === 1 ? "" : "s"}${freed}.`
      : "Nothing cached to forget.", "ok");
    if (res.kept?.length) {
      toast(`Still in use, kept: ${res.kept.join(", ")}`, "error");
    }
    await loadLibrary();
  } catch (err) {
    toast(`Could not clear the cache: ${err.message}`, "error");
  } finally {
    button.disabled = false;
    disarm(button);
  }
});

/** How much of your disk this list is, so "forget" has a visible payoff. */
function renderLibraryNote(items) {
  const note = $("#library-note");
  const bytes = items.reduce((sum, i) => sum + (i.size || 0), 0);
  note.textContent = items.length
    ? `${items.length} transcript${items.length === 1 ? "" : "s"} cached in work/, ${fmtSize(bytes)}.`
    : "";
  $("#btn-forget-all").classList.toggle("hidden", !items.length);
}

/* ------------------------------------------------------- two-click confirming

   Same shape everywhere something destructive is one click away: the first click
   states what will happen, and the button disarms itself after four seconds,
   because a button left armed is a trap for whoever comes back to it.          */

function arm(button, question) {
  if (button.dataset.confirming === "yes") return true;
  button.dataset.confirming = "yes";
  button.dataset.label = button.textContent;
  button.textContent = question;
  button.classList.add("danger");
  setTimeout(() => disarm(button), 4000);
  return false;
}

function disarm(button) {
  if (button.dataset.confirming !== "yes") return;
  button.dataset.confirming = "";
  button.textContent = button.dataset.label || button.textContent;
  button.classList.remove("danger");
}

/* ======================================================= translate via Google */

/* -------------------------------------------- online providers (Google/DeepL) */

async function loadProviders() {
  let info;
  try {
    info = await api("/api/translate/providers");
  } catch {
    return;
  }
  state.providers = info;
  $("#keys-path").textContent = info.keys_path;
  // A key from the environment cannot be changed here; say so rather than let
  // "Save keys" look like it did nothing.
  const fromEnv = Object.entries(info.key_source || {})
    .filter(([, src]) => src.startsWith("SGEN_"))
    .map(([name, src]) => `${name} comes from ${src}`);
  $("#keys-source").textContent = fromEnv.length
    ? `${fromEnv.join("; ")} — saving here will not override that.`
    : "";

  // The same key state, shown in the Settings dropdown, so "Cloud: DeepL" does
  // not look available when it cannot run.
  for (const [value, label] of [["google", "Cloud: Google Translate"],
                                ["deepl", "Cloud: DeepL"]]) {
    const option = $("#opt-translate-mode").querySelector?.(`option[value="${value}"]`);
    if (option) {
      option.textContent = label + (info.configured[value] ? "" : " — needs a key");
    }
  }
  updateTranslateMode();

  const options = [
    ["google", "Google Translate", info.configured.google],
    ["deepl", "DeepL", info.configured.deepl],
  ];
  $("#auto-provider").innerHTML = options
    .map(([value, label, ready]) =>
      `<option value="${value}">${label}${ready ? "" : " — needs a key"}</option>`)
    .join("");
  // Prefer the configured provider, but only if it has a key; otherwise pick
  // whichever one is actually usable.
  const preferred = state.meta?.defaults?.translate_provider;
  const ready = options.find(([v, , r]) => r && v === preferred)
    || options.find(([, , r]) => r);
  if (ready) $("#auto-provider").value = ready[0];

  const target = state.meta?.defaults?.translate_target || "en";
  fillTargets($("#auto-lang"), $("#auto-provider").value, target);

  updateAutoHint();
  // If nothing is configured, open the key panel so the next step is obvious.
  $("#keys-block").open = !info.configured.google && !info.configured.deepl;
}

function updateAutoHint() {
  const provider = $("#auto-provider").value;
  const lang = $("#auto-lang").value;
  const info = state.providers;
  if (!info) return;

  const ready = info.configured[provider];
  const parts = [];
  if (!ready) parts.push(`No ${provider} key yet — add one under “API keys”.`);
  if (provider === "deepl" && !info.deepl_targets.includes(lang)) {
    parts.push(`DeepL cannot translate into ${lang} — use Google.`);
  }
  parts.push("Sends the transcript text to the service.");
  $("#auto-hint").textContent = parts.join(" ");
  $("#btn-auto-translate").disabled = !ready;
}

$("#auto-provider").addEventListener("change", () => {
  // Switching provider changes which languages are reachable.
  fillTargets($("#auto-lang"), $("#auto-provider").value, $("#auto-lang").value);
  updateAutoHint();
});
$("#auto-lang").addEventListener("change", updateAutoHint);

$("#btn-save-keys").addEventListener("click", async () => {
  const body = { deepl_plan: $("#key-deepl-plan").value };
  const google = $("#key-google").value.trim();
  const deepl = $("#key-deepl").value.trim();
  if (google) body.google = google;
  if (deepl) body.deepl = deepl;
  try {
    const res = await api("/api/translate/keys", {
      method: "POST", body: JSON.stringify(body),
    });
    $("#key-google").value = "";
    $("#key-deepl").value = "";
    $("#keys-status").textContent = `saved to ${res.path}`;
    toast("Keys saved.", "ok");
    await loadProviders();
  } catch (err) {
    toast(`Could not save keys: ${err.message}`, "error");
  }
});

$("#btn-test-key").addEventListener("click", async () => {
  const provider = $("#auto-provider").value;
  $("#keys-status").textContent = "testing…";
  try {
    const res = await api("/api/translate/test", {
      method: "POST", body: JSON.stringify({ provider }),
    });
    $("#keys-status").textContent = `${provider} works — "hello" → "${res.sample}"`;
    toast(`${provider} key is working.`, "ok");
  } catch (err) {
    $("#keys-status").textContent = "";
    toast(`${provider}: ${err.message}`, "error");
  }
});

$("#btn-auto-translate").addEventListener("click", async () => {
  if (!state.translate.contentId) return;
  const btn = $("#btn-auto-translate");
  btn.disabled = true;
  $("#auto-status").textContent = "translating…";
  try {
    const res = await api(`/api/result/${state.translate.contentId}/translate-online`, {
      method: "POST",
      body: JSON.stringify({
        provider: $("#auto-provider").value,
        language: $("#auto-lang").value,
        formats: ($("#fmt-srt").checked ? ["srt"] : []).concat(
          $("#fmt-vtt").checked ? ["vtt"] : []),
        out_dir: $("#opt-outdir").value.trim() || null,
      }),
    });
    $("#auto-status").textContent =
      `${res.cue_count} subtitles · ${res.characters.toLocaleString()} characters`;
    $("#ext-result").textContent = `Wrote ${res.written.length} file(s).`;
    toast(`Translated with ${res.provider}: ${res.cue_count} subtitles.`, "ok");
    loadLibrary();
  } catch (err) {
    $("#auto-status").textContent = "";
    toast(`Translation failed: ${err.message}`, "error");
  } finally {
    updateAutoHint();
  }
});

function openTranslate(contentId, name) {
  state.translate = { contentId, name };
  $("#translate-target-name").textContent = name;
  $("#ext-input").value = "";
  $("#ext-result").textContent = "";
  $("#copy-status").textContent = "";
  $("#auto-status").textContent = "";
  updateAutoHint();
  // The panel lives on the Translate tab, so a press from the Subtitles tab
  // has to take you there — otherwise it opens somewhere you cannot see.
  showView("translate");
  const panel = $("#translate-panel");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "center" });
}

$("#btn-close-translate").addEventListener("click", () => {
  $("#translate-panel").classList.add("hidden");
  state.translate = { contentId: null, name: "" };
});

$("#btn-copy-text").addEventListener("click", async () => {
  if (!state.translate.contentId) return;
  try {
    const data = await api(`/api/result/${state.translate.contentId}/export-text`);
    try {
      await navigator.clipboard.writeText(data.text);
      $("#copy-status").textContent =
        `${data.cue_count} lines copied — off this machine once you paste them out`;
      toast(`Copied ${data.cue_count} numbered lines.`, "ok");
    } catch {
      // Clipboard needs a secure context; put the text where it can be copied.
      $("#ext-input").value = data.text;
      $("#ext-input").select();
      $("#copy-status").textContent =
        "clipboard blocked — text placed in the box below, copy it from there";
    }
  } catch (err) {
    toast(`Export failed: ${err.message}`, "error");
  }
});

$("#btn-apply-translation").addEventListener("click", async () => {
  if (!state.translate.contentId) return;
  const text = $("#ext-input").value.trim();
  if (!text) return toast("Paste the translated text first.", "error");

  const formats = [];
  if ($("#fmt-srt").checked) formats.push("srt");
  if ($("#fmt-vtt").checked) formats.push("vtt");

  try {
    const res = await api(`/api/result/${state.translate.contentId}/import-translation`, {
      method: "POST",
      body: JSON.stringify({
        text,
        language: $("#ext-lang").value.trim() || "en",
        formats: formats.length ? formats : ["srt", "vtt"],
        keep_untranslated: true,
        out_dir: $("#opt-outdir").value.trim() || null,
      }),
    });
    $("#ext-result").textContent =
      `${res.summary} → ${res.written.length} file(s) written.`;
    toast(`${res.matched}/${res.total} lines applied.`,
          res.missing.length ? "" : "ok");
  } catch (err) {
    $("#ext-result").textContent = "";
    toast(`Import failed: ${err.message}`, "error");
  }
});

/* ====================================================================== tune */

const TUNE_SPEC = [
  { group: "gating", key: "min_mean_word_prob", min: 0, max: 1, step: 0.01,
    hint: "Word-confidence floor. The first knob to relax if real speech is being cut." },
  { group: "gating", key: "hard_no_speech_prob", min: 0, max: 1, step: 0.01,
    hint: "Suppress on its own past this. Raise it to keep more." },
  { group: "gating", key: "hard_avg_logprob", min: -3, max: 0, step: 0.05,
    hint: "Suppress on its own below this. Lower keeps more." },
  { group: "gating", key: "max_compression_ratio", min: 1, max: 5, step: 0.05,
    hint: "Repetition detector. Lower is stricter." },
  { group: "gating", key: "min_words_per_second", min: 0, max: 2, step: 0.05,
    hint: "Below this over a long stretch reads as text smeared over non-speech." },
  { group: "cues", key: "max_chars_per_line", min: 20, max: 60, step: 1,
    hint: "42 is the broadcast norm." },
  { group: "cues", key: "target_cps", min: 8, max: 25, step: 0.5,
    hint: "Reading speed. 17 is the usual adult figure." },
];

function buildTuneControls(defaults) {
  $("#tune-controls").innerHTML = TUNE_SPEC.map((spec) => {
    const value = defaults[spec.group][spec.key];
    state.tune.values[`${spec.group}.${spec.key}`] = value;
    return `<div class="slider">
      <div class="slider-head">
        <label for="t-${spec.key}">${spec.key}</label>
        <output id="o-${spec.key}">${value}</output>
      </div>
      <input type="range" id="t-${spec.key}" min="${spec.min}" max="${spec.max}"
        step="${spec.step}" value="${value}"
        data-group="${spec.group}" data-key="${spec.key}">
      <p class="hint">${spec.hint}</p>
    </div>`;
  }).join("");
}

$("#tune-controls").addEventListener("input", (event) => {
  const input = event.target;
  if (input.type !== "range") return;
  $(`#o-${input.dataset.key}`).textContent = input.value;
  state.tune.values[`${input.dataset.group}.${input.dataset.key}`] = +input.value;
  clearTimeout(state.tune.timer);
  state.tune.timer = setTimeout(runRegate, 220);
});

$("#tune-job").addEventListener("change", (event) => {
  state.tune.contentId = event.target.value || null;
  if (state.tune.contentId) runRegate();
});

$("#btn-tune-reset").addEventListener("click", () => {
  buildTuneControls(state.meta.defaults);
  if (state.tune.contentId) runRegate();
});

$("#btn-tune-apply").addEventListener("click", async () => {
  try {
    await api(`/api/profile/${$("#opt-profile").value}`, {
      method: "POST",
      body: JSON.stringify(collectTuneValues()),
    });
    toast("Saved. New runs will use these thresholds.", "ok");
    state.meta = await api("/api/meta");
  } catch (err) {
    toast(`Could not save: ${err.message}`, "error");
  }
});

function collectTuneValues() {
  const gating = {}, cues = {};
  for (const [path, value] of Object.entries(state.tune.values)) {
    const [group, key] = path.split(".");
    (group === "gating" ? gating : cues)[key] = value;
  }
  return { gating, cues };
}

function refreshTunePicker() {
  // Draw from the on-disk library as well, or a restart empties this too.
  const seen = new Map();
  for (const job of state.jobs.values()) {
    if (job.status === "done" && job.content_id) seen.set(job.content_id, job.name);
  }
  for (const item of state.library) {
    if (!seen.has(item.content_id)) seen.set(item.content_id, item.name);
  }

  const select = $("#tune-job");
  const current = select.value;
  select.innerHTML = [`<option value="">— pick a file —</option>`]
    .concat(Array.from(seen, ([id, name]) =>
      `<option value="${id}">${escapeHtml(name)}</option>`))
    .join("");
  if (seen.has(current)) select.value = current;
}

async function runRegate() {
  if (!state.tune.contentId) return;
  let data;
  try {
    data = await api(`/api/result/${state.tune.contentId}/regate`, {
      method: "POST",
      body: JSON.stringify({ ...collectTuneValues(), keep_suppressed: false }),
    });
  } catch (err) {
    return toast(`Re-check failed: ${err.message}`, "error");
  }

  $("#btn-tune-apply").disabled = false;
  const { stats } = data;
  $("#tune-stats").innerHTML = [
    `<div class="stat good"><div class="k">kept</div><div class="v">${stats.kept}</div></div>`,
    `<div class="stat cut"><div class="k">suppressed</div><div class="v">${stats.suppressed}</div></div>`,
    `<div class="stat"><div class="k">subtitles</div><div class="v">${data.cues.length}</div></div>`,
    ...Object.entries(stats.reasons).sort((a, b) => b[1] - a[1]).map(([r, n]) =>
      `<div class="stat"><div class="k">${escapeHtml(r)}</div><div class="v">${n}</div></div>`),
  ].join("");

  const suppressed = data.segments.filter((s) => s.suppressed);
  $("#tune-suppressed").innerHTML = suppressed.length
    ? suppressed.map((s) => `<div class="sup-item">
        <div class="meta">
          <span>${s.start.toFixed(1)}s</span>
          <span class="reason">${escapeHtml(s.suppress_reason || "")}</span>
          <span>p=${(s.mean_word_prob ?? 0).toFixed(2)}</span>
        </div>
        <div class="txt">${escapeHtml(s.text || "")}</div>
      </div>`).join("")
    : `<p class="muted">Nothing suppressed at these thresholds.</p>`;
}

/* ====================================================================== boot */

/**
 * Start each part independently.
 *
 * Chained .then() meant the first rejection skipped everything after it, so a
 * server that was restarting for one second left a page where unrelated panels
 * did nothing. Each step now fails alone and says so.
 */
async function boot() {
  const steps = [
    ["file browser", initBrowser],   // first: it is what the page is for
    ["settings", loadMeta],
    ["translators", loadProviders],
    ["library", loadLibrary],
    ["tune list", refreshTunePicker],
  ];
  for (const [label, step] of steps) {
    try {
      await step();
    } catch (err) {
      console.error(`sgen: ${label} failed`, err);
      toast(`${label} failed to load: ${err.message}`, "error");
    }
  }
  connectEvents();
}

boot();
