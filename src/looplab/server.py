"""Local UI server: pick a video, configure, analyze, explore.

`looplab --ui` serves a localhost-only explorer shell. The flow is staged:
"Open video..." asks the *server* to raise the native OS file dialog
(macOS: osascript `choose file`; elsewhere: tkinter), so looplab gets a
real filesystem path and reads the original file in place - no upload, no
copy. Opening a video analyzes nothing: the settings dropdown holds the
tuning parameters, the attention crop (a rectangle the seam *search* is
restricted to - rendered loops stay full-frame), ignore time ranges, and
the Analyze button. Analysis streams a weighted progress bar and can be
stopped mid-run. Existing results (`<video>.looplab/`) are never loaded
automatically - a "Load previous results" button appears when they exist.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import core, explorer

# stage -> (start, end) share of the overall progress bar
_WEIGHTS = {"decode": (0.0, 0.12), "score": (0.12, 0.55),
            "post": (0.55, 0.58), "render": (0.58, 1.0)}

# parameters the UI may set, with coercion
_PARAM_TYPES = {"min_loop": float, "max_loop": float, "window": int,
                "min_activity": float, "focus_weight": float,
                "vel_weight": float}


class _Cancelled(Exception):
    """Raised inside an analysis run when the user pressed Stop."""


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "idle"  # idle | loaded | analyzing | ready | error
        self.stage = ""
        self.pct = 0.0
        self.log: deque[str] = deque(maxlen=60)
        self.workdir: Path | None = None
        self.video: str | None = None
        self.info: dict | None = None         # ffprobe of the loaded video
        self.result_info: dict | None = None  # fps/n_frames of loaded results
        self.error: str | None = None
        self.run = 0                          # token: stale threads stand down
        self.cancel = threading.Event()


STATE = _State()


def _workdir_of(path: str) -> Path:
    src = Path(path)
    return src.with_suffix(src.suffix + ".looplab")


def _has_previous(path: str) -> bool:
    return (_workdir_of(path) / "index.html").exists()


def _manifest_info(wd: Path) -> dict | None:
    try:
        m = json.loads((wd / "candidates.json").read_text())
        return {"fps": m.get("fps"), "n_frames": m.get("n_frames")}
    except Exception:
        return None


def _progress_for(run: int, cancel: threading.Event):
    def progress(stage: str, frac: float) -> None:
        if cancel.is_set():
            raise _Cancelled()
        lo, hi = _WEIGHTS.get(stage, (0.0, 0.0))
        pct = (lo + (hi - lo) * min(max(frac, 0.0), 1.0)) * 100.0
        with STATE.lock:
            if STATE.run == run:
                STATE.stage = stage
                STATE.pct = round(max(STATE.pct, pct), 1)  # monotonic
    return progress


def _coerce_params(raw: dict) -> core.SearchParams:
    kw = {}
    for key, cast in _PARAM_TYPES.items():
        if raw.get(key) not in (None, "", "auto"):
            try:
                kw[key] = cast(raw[key])
            except (TypeError, ValueError):
                pass
    p = raw.get("proxy_long")
    if p not in (None, "", "auto"):
        try:
            kw["proxy_long"] = int(p)
        except (TypeError, ValueError):
            pass
    c = raw.get("crop")
    if (isinstance(c, (list, tuple)) and len(c) == 4
            and all(isinstance(v, (int, float)) for v in c)
            and c[2] > 0.01 and c[3] > 0.01
            and 0 <= c[0] and 0 <= c[1]
            and c[0] + c[2] <= 1.0001 and c[1] + c[3] <= 1.0001):
        kw["crop"] = tuple(round(float(v), 4) for v in c)
    ig = raw.get("ignore")
    if isinstance(ig, (list, tuple)):
        ranges = []
        for pair in ig:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                try:
                    lo, hi = float(pair[0]), float(pair[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= lo < hi:
                    ranges.append((round(lo, 3), round(hi, 3)))
        if ranges:
            kw["ignore_ranges"] = ranges
    return core.SearchParams(**kw)


def _open_video(path: str) -> dict:
    """Probe and stage a video for analysis. Loads no results, runs nothing."""
    info = core.ffprobe_video(path)
    with STATE.lock:
        STATE.video, STATE.info = path, info
        STATE.phase, STATE.stage, STATE.pct = "loaded", "", 0.0
        STATE.error, STATE.workdir, STATE.result_info = None, None, None
        STATE.log.clear()
    return info


def _adopt(path: str) -> None:
    """Serve the existing <video>.looplab/ results without re-analyzing."""
    wd = _workdir_of(path)
    info = core.ffprobe_video(path)
    ri = _manifest_info(wd)
    with STATE.lock:
        STATE.video, STATE.info, STATE.workdir = path, info, wd
        STATE.result_info = ri
        STATE.phase, STATE.stage, STATE.pct = "ready", "", 100.0
        STATE.error = None


def _analyze(path: str, raw_params: dict | None = None) -> None:
    with STATE.lock:
        STATE.run += 1
        run = STATE.run
        STATE.cancel = threading.Event()
        cancel = STATE.cancel
        STATE.phase, STATE.video, STATE.error = "analyzing", path, None
        STATE.stage, STATE.pct = "decode", 0.0
        STATE.log.clear()
    progress = _progress_for(run, cancel)

    def log(msg: str) -> None:
        with STATE.lock:
            if STATE.run == run:
                STATE.log.append(msg)

    try:
        params = _coerce_params(raw_params or {})
        src = Path(path)
        result = core.run_search(str(src), params, log=log, progress=progress)
        workdir = src.with_suffix(src.suffix + ".looplab")
        core.save_workdir(result, workdir)
        if not result.candidates:
            raise RuntimeError("no seam survived the gates - lower min "
                               "activity or widen the loop range in settings")
        log("[render] cutting candidate loops + scrub proxy...")
        explorer.render_explorer_assets(
            result, workdir,
            progress=lambda f: progress("render", f))
        explorer.build(result, workdir, mode="local")
        explorer.build(result, workdir, mode="artifact")
        progress("render", 1.0)  # last cancellation point before adopting
        with STATE.lock:
            if STATE.run == run:
                STATE.workdir, STATE.phase, STATE.pct = workdir, "ready", 100.0
                STATE.result_info = {"fps": result.fps,
                                     "n_frames": result.n_frames}
    except _Cancelled:
        with STATE.lock:
            if STATE.run == run:
                # fall back to the still-loaded old results if we had some
                STATE.phase = "ready" if STATE.workdir else "loaded"
                STATE.stage, STATE.pct = "", 0.0
                STATE.log.append("[stop] analysis stopped")
    except Exception as e:
        with STATE.lock:
            if STATE.run == run:
                STATE.phase, STATE.error = "error", str(e)


def _pick_file() -> str | None:
    """Raise the native OS open-file dialog and return the chosen path."""
    if sys.platform == "darwin":
        r = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose file with prompt '
             '"Choose a video for looplab" of type {"public.movie"})'],
            capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.update()
        p = filedialog.askopenfilename(title="Choose a video for looplab")
        root.destroy()
        return p or None
    except Exception as e:  # headless box, no tk - CLI still works
        raise RuntimeError(f"no native file dialog available ({e}); "
                           "run: looplab <video>  then  looplab --ui") from e


def _poster_frame(path: str, t: float) -> bytes:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(max(t, 0.0)), "-i", path,
         "-frames:v", "1", "-vf", "scale=560:-2", "-f", "image2pipe",
         "-vcodec", "mjpeg", "pipe:1"],
        capture_output=True, check=True)
    return r.stdout


TOOLBAR = """
<div id="ll-toolbar" style="position:sticky;top:0;z-index:50;background:#141217f2;
  border-bottom:1px solid #322e38;font:13px/1.4 system-ui,sans-serif;color:#ece7df;
  backdrop-filter:blur(4px)">
  <div style="display:flex;gap:12px;align-items:center;padding:9px 24px;flex-wrap:wrap">
    <strong style="letter-spacing:.06em">looplab</strong>
    <button id="ll-open" class="ll-btn ll-accent">Open video&hellip;</button>
    <button id="ll-params" class="ll-btn ll-icon" title="Analysis settings"
      aria-label="Analysis settings" aria-expanded="false">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true">
        <line x1="3" y1="6" x2="21" y2="6"/><circle cx="9" cy="6" r="2.7"/>
        <line x1="3" y1="12" x2="21" y2="12"/><circle cx="15" cy="12" r="2.7"/>
        <line x1="3" y1="18" x2="21" y2="18"/><circle cx="7" cy="18" r="2.7"/>
      </svg>
    </button>
    <button id="ll-stop" class="ll-btn ll-stop" hidden>Stop</button>
    <span id="ll-status" style="color:#8f8798;font-family:ui-monospace,Menlo,monospace;
      font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      max-width:52ch"></span>
  </div>
  <div id="ll-panel">
    <div class="ll-row">
      <label class="ll-f">min loop (s)<input id="p-min_loop" type="number" step="0.1" min="0.1"></label>
      <label class="ll-f">max loop (s)<input id="p-max_loop" type="number" step="0.5" min="0.2"></label>
      <label class="ll-f">proxy px<select id="p-proxy_long">
        <option value="auto">auto</option><option>512</option><option>384</option>
        <option>256</option><option>192</option></select></label>
      <label class="ll-f">seam window (&plusmn;f)<input id="p-window" type="number" step="1" min="1"></label>
      <label class="ll-f">min activity<input id="p-min_activity" type="number" step="0.1" min="0"></label>
      <label class="ll-f">focus weight<input id="p-focus_weight" type="number" step="0.5" min="0"></label>
      <label class="ll-f">velocity weight<input id="p-vel_weight" type="number" step="0.5" min="0"></label>
    </div>
    <div class="ll-row">
      <div class="ll-sec">
        <span class="ll-sechead">attention crop</span>
        <div style="display:flex;gap:8px;align-items:center">
          <button id="ll-crop" class="ll-btn">Set&hellip;</button>
          <button id="ll-cropclear" class="ll-btn" hidden>Clear</button>
          <span id="ll-cropval" class="ll-mono">full frame</span>
        </div>
        <span class="ll-hint">the seam search only looks inside it; loops render full-frame</span>
      </div>
      <div class="ll-sec" style="flex:1;min-width:280px">
        <span class="ll-sechead">ignore time ranges (s)</span>
        <input id="p-ignore" type="text" placeholder="e.g. 0-4.5, 42-47"
          spellcheck="false" autocomplete="off">
        <span class="ll-hint">no loop will overlap these times &middot; tip: shift-drag the heatmap</span>
      </div>
    </div>
    <div id="ll-croparea" hidden>
      <canvas id="ll-cropcv"></canvas>
      <span class="ll-hint">drag a rectangle on the frame; drag a tiny one (or Clear) to remove</span>
    </div>
    <div class="ll-row" style="margin-bottom:0">
      <button id="ll-analyze" class="ll-btn ll-accent" disabled>Analyze</button>
      <button id="ll-loadprev" class="ll-btn" hidden>Load previous results</button>
      <button id="ll-reset" class="ll-btn">Reset</button>
    </div>
  </div>
  <div style="height:3px;background:#241f2a">
    <div id="ll-bar" style="height:100%;width:0%;background:#e07a63;transition:width .4s"></div>
  </div>
</div>
<style>
  .ll-btn { background:#2a2630; border:1px solid #4a4452; border-radius:6px; color:#ece7df;
    padding:6px 13px; font:600 12px system-ui; cursor:pointer; }
  .ll-btn:hover { border-color:#e07a63; }
  .ll-btn[disabled] { opacity:.55; cursor:default; }
  .ll-accent { background:#e07a63; border-color:#e07a63; color:#fff; }
  .ll-stop { background:#7a2f2f; border-color:#a84444; }
  .ll-stop:hover { border-color:#e05656; }
  .ll-f { display:flex; flex-direction:column; gap:4px; font-size:11px; color:#8f8798;
    text-transform:uppercase; letter-spacing:.08em; }
  .ll-f input, .ll-f select { width:96px; background:#1d1a21; color:#ece7df;
    border:1px solid #4a4452; border-radius:5px; padding:5px 7px; font:13px ui-monospace,Menlo,monospace; }
  .ll-icon { display:inline-flex; align-items:center; padding:6px 9px; }
  .ll-icon.on { background:#e07a63; border-color:#e07a63; color:#fff; }
  .ll-row { display:flex; gap:18px; flex-wrap:wrap; align-items:end; margin:0 0 14px; }
  .ll-sechead { font-size:11px; color:#8f8798; text-transform:uppercase;
    letter-spacing:.08em; display:block; margin:0 0 5px; }
  .ll-hint { font-size:11px; color:#6f6879; display:block; margin-top:5px; }
  .ll-mono { font-family:ui-monospace,Menlo,monospace; font-size:12px; color:#8f8798; }
  #p-ignore { width:100%; box-sizing:border-box; background:#1d1a21; color:#ece7df;
    border:1px solid #4a4452; border-radius:5px; padding:6px 8px;
    font:13px ui-monospace,Menlo,monospace; }
  #p-ignore.ll-bad { border-color:#e05656; }
  #ll-panel { display:none; padding:12px 24px 16px; border-top:1px solid #2a2630; }
  #ll-panel.open { display:block; animation:llslide .16s ease-out; }
  #ll-croparea { margin:0 0 14px; }
  #ll-cropcv { display:block; border-radius:6px; cursor:crosshair; max-width:560px;
    width:100%; height:auto; }
  @keyframes llslide { from { opacity:0; transform:translateY(-5px); } }
  @media (prefers-reduced-motion: reduce) { #ll-panel.open { animation:none; } }
</style>
<script>
(function () {
  const $ = id => document.getElementById(id);
  const openBtn = $('ll-open'), st = $('ll-status'), bar = $('ll-bar');
  const analyzeBtn = $('ll-analyze'), stopBtn = $('ll-stop');
  const loadPrevBtn = $('ll-loadprev'), ignoreEl = $('p-ignore');
  const P_KEYS = ['min_loop','max_loop','proxy_long','window','min_activity',
                  'focus_weight','vel_weight'];
  const DEFAULTS = __DEFAULTS__;
  let video = null, hasPrev = false, phase = 'idle', dur = 0;
  let llCrop = null, posterImg = null;
  const clamp01 = x => Math.min(Math.max(x, 0), 1);
  const post = (url, body) => fetch(url, { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}) }).then(r => r.json());

  // ---- storage ----
  const jget = (k, d) => {
    try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; }
  };
  const jset = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  function loadParams() {
    const saved = jget('ll-params', {});
    for (const k of P_KEYS) { const el = $('p-' + k); if (el) el.value = saved[k] ?? DEFAULTS[k]; }
  }
  function loadVideoPrefs() {
    const v = video ? jget('ll-v:' + video, {}) : {};
    llCrop = Array.isArray(v.crop) ? v.crop : null;
    ignoreEl.value = v.ignore || '';
    updateCropUI();
  }
  function saveVideoPrefs() {
    if (video) jset('ll-v:' + video, { crop: llCrop, ignore: ignoreEl.value });
  }
  function saveRecent(path) {
    if (!path) return;
    const r = jget('ll-recents', []).filter(e => e.path !== path);
    r.unshift({ path: path, name: path.split('/').pop() });
    jset('ll-recents', r.slice(0, 6));
  }

  // ---- params ----
  function parseIgnore() {
    ignoreEl.classList.remove('ll-bad');
    const txt = ignoreEl.value.trim();
    if (!txt) return [];
    const out = [];
    for (const tok of txt.split(/[,;]+/)) {
      if (!tok.trim()) continue;
      const m = tok.trim().match(/^(\\d+(?:\\.\\d+)?)\\s*[-:]\\s*(\\d+(?:\\.\\d+)?)$/);
      if (!m || !(parseFloat(m[2]) > parseFloat(m[1]))) {
        ignoreEl.classList.add('ll-bad');
        return null;
      }
      out.push([parseFloat(m[1]), parseFloat(m[2])]);
    }
    return out;
  }
  function gatherParams() {
    const out = {};
    for (const k of P_KEYS) { const el = $('p-' + k); if (el) out[k] = el.value; }
    jset('ll-params', out);
    const ig = parseIgnore();
    if (ig === null) return null;
    if (ig.length) out.ignore = ig;
    if (llCrop) out.crop = llCrop;
    saveVideoPrefs();
    return out;
  }
  $('ll-reset').addEventListener('click', () => {
    localStorage.removeItem('ll-params');
    if (video) localStorage.removeItem('ll-v:' + video);
    loadParams();
    loadVideoPrefs();
  });
  ignoreEl.addEventListener('change', () => { parseIgnore(); saveVideoPrefs(); });

  function setPanel(open) {
    $('ll-panel').classList.toggle('open', open);
    $('ll-params').classList.toggle('on', open);
    $('ll-params').setAttribute('aria-expanded', String(open));
  }
  $('ll-params').addEventListener('click',
    () => setPanel(!$('ll-panel').classList.contains('open')));

  // ---- attention crop ----
  const cv = $('ll-cropcv');
  let dragging = null;
  function updateCropUI() {
    $('ll-cropval').textContent = llCrop
      ? 'x=' + llCrop[0].toFixed(3) + ' y=' + llCrop[1].toFixed(3)
        + ' w=' + llCrop[2].toFixed(3) + ' h=' + llCrop[3].toFixed(3)
      : 'full frame';
    $('ll-cropclear').hidden = !llCrop;
    drawCrop();
  }
  function drawCrop() {
    if (!posterImg) return;
    const ctx = cv.getContext('2d');
    ctx.drawImage(posterImg, 0, 0, cv.width, cv.height);
    if (llCrop) {
      const x = llCrop[0], y = llCrop[1], w = llCrop[2], h = llCrop[3];
      ctx.fillStyle = 'rgba(0,0,0,.55)';
      ctx.fillRect(0, 0, cv.width, y * cv.height);
      ctx.fillRect(0, (y + h) * cv.height, cv.width, cv.height);
      ctx.fillRect(0, y * cv.height, x * cv.width, h * cv.height);
      ctx.fillRect((x + w) * cv.width, y * cv.height, cv.width, h * cv.height);
      ctx.strokeStyle = '#e07a63'; ctx.lineWidth = 2;
      ctx.strokeRect(x * cv.width, y * cv.height, w * cv.width, h * cv.height);
    }
  }
  $('ll-crop').addEventListener('click', () => {
    if (!video) { st.textContent = 'open a video first'; return; }
    const area = $('ll-croparea');
    area.hidden = !area.hidden;
    if (area.hidden) return;
    if (posterImg) { drawCrop(); return; }
    const img = new Image();
    img.onload = () => {
      posterImg = img;
      cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      drawCrop();
    };
    img.src = '/frame?path=' + encodeURIComponent(video) + '&t=1';
  });
  $('ll-cropclear').addEventListener('click', () => {
    llCrop = null; updateCropUI(); saveVideoPrefs();
  });
  const cvPos = ev => {
    const r = cv.getBoundingClientRect();
    return [clamp01((ev.clientX - r.left) / r.width),
            clamp01((ev.clientY - r.top) / r.height)];
  };
  cv.addEventListener('mousedown', ev => { dragging = cvPos(ev); ev.preventDefault(); });
  cv.addEventListener('mousemove', ev => {
    if (!dragging) return;
    const a = dragging, b = cvPos(ev);
    llCrop = [Math.min(a[0], b[0]), Math.min(a[1], b[1]),
              Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])];
    updateCropUI();
  });
  addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = null;
    if (llCrop && (llCrop[2] < 0.02 || llCrop[3] < 0.02)) llCrop = null;
    updateCropUI();
    saveVideoPrefs();
  });

  // ---- status / flow ----
  function videoLine(s) {
    if (!s.video) return '';
    const name = s.video.split('/').pop();
    const i = s.result_info || s.info || {};
    const nf = i.n_frames || i.nb_frames;
    if (nf && i.fps)
      return name + ' \\u00b7 ' + (nf / i.fps).toFixed(1) + 's \\u00b7 '
        + nf + 'f @ ' + (+i.fps).toFixed(2);
    if (i.duration) return name + ' \\u00b7 ' + (+i.duration).toFixed(1) + 's';
    return name;
  }
  function apply(s) {
    phase = s.phase; video = s.video || null; hasPrev = !!s.has_previous;
    const i = s.result_info || s.info;
    dur = (i && i.n_frames && i.fps) ? i.n_frames / i.fps
        : (s.info && +s.info.duration) || 0;
    analyzeBtn.disabled = !video || phase === 'analyzing';
    analyzeBtn.textContent = phase === 'ready' ? 'Re-analyze' : 'Analyze';
    loadPrevBtn.hidden = !(hasPrev && phase !== 'ready' && phase !== 'analyzing');
    stopBtn.hidden = phase !== 'analyzing';
    openBtn.disabled = phase === 'analyzing';
    if (phase === 'loaded')
      st.textContent = videoLine(s)
        + (hasPrev ? ' \\u00b7 previous results available' : '');
    else if (phase === 'ready') st.textContent = 'ready \\u00b7 ' + videoLine(s);
    else if (phase === 'error')
      st.textContent = 'error: ' + (s.error || 'analysis failed');
  }
  const STAGE_LABEL = { decode: 'decoding', score: 'scoring seams',
                        post: 'gating', render: 'rendering previews' };
  let polling = false, stopped = false;
  async function poll() {
    polling = true;
    const s = await (await fetch('/status')).json();
    if (s.phase === 'analyzing') {
      bar.style.width = s.pct + '%';
      stopBtn.hidden = false; analyzeBtn.disabled = true; openBtn.disabled = true;
      st.textContent = (STAGE_LABEL[s.stage] || s.stage) + ' ' + s.pct + '%'
        + (s.log && s.log.length ? ' \\u2014 ' + s.log[s.log.length - 1] : '');
      setTimeout(poll, 600);
      return;
    }
    polling = false;
    if (s.phase === 'ready' && !stopped) {
      bar.style.width = '100%';
      saveRecent(s.video);
      st.textContent = 'ready \\u2014 loading explorer\\u2026';
      location.reload();
      return;
    }
    stopped = false;
    bar.style.width = '0%';
    apply(s);
    if (s.phase === 'ready')
      st.textContent = 'analysis stopped \\u00b7 kept previous results';
    else if (s.phase === 'loaded')
      st.textContent = 'analysis stopped \\u00b7 ' + videoLine(s);
  }
  analyzeBtn.addEventListener('click', async () => {
    if (!video) return;
    const params = gatherParams();
    if (params === null) {
      st.textContent = 'fix the ignore ranges field (e.g. 0-4.5, 42-47)';
      setPanel(true);
      return;
    }
    analyzeBtn.disabled = true; openBtn.disabled = true;
    bar.style.width = '0%';
    const r = await post('/analyze', { path: video, params: params });
    if (!r.ok) {
      st.textContent = 'error: ' + (r.error || 'could not start');
      analyzeBtn.disabled = false; openBtn.disabled = false;
      return;
    }
    st.textContent = 'analyzing ' + video.split('/').pop() + '\\u2026';
    stopBtn.hidden = false;
    if (!polling) poll();
  });
  stopBtn.addEventListener('click', async () => {
    stopBtn.disabled = true;
    stopped = true;
    st.textContent = 'stopping\\u2026';
    await post('/cancel');
    setTimeout(() => { stopBtn.disabled = false; }, 1500);
  });
  loadPrevBtn.addEventListener('click', async () => {
    const r = await post('/load', { path: video });
    if (r.ok) { saveRecent(video); location.reload(); }
    else st.textContent = 'error: ' + (r.error || 'could not load previous results');
  });
  openBtn.addEventListener('click', async () => {
    openBtn.disabled = true;
    st.textContent = 'choose a file in the dialog\\u2026';
    try {
      const r = await post('/pick');
      if (!r.path) {
        openBtn.disabled = false;
        st.textContent = r.error ? 'error: ' + r.error : '';
        return;
      }
      location.reload();
    } catch (e) { st.textContent = 'error: ' + e; openBtn.disabled = false; }
  });

  // ---- recent videos (landing page only) ----
  function renderRecents() {
    const box = $('ll-recents');
    if (!box) return;
    const recents = jget('ll-recents', []);
    $('ll-recents-wrap').hidden = !recents.length;
    box.textContent = '';
    for (const e of recents) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:10px;align-items:center;margin:6px 0';
      const b = document.createElement('button');
      b.className = 'll-btn';
      b.textContent = e.name;
      b.addEventListener('click', async () => {
        const r = await post('/open', { path: e.path });
        if (r.ok) { location.reload(); return; }
        st.textContent = 'error: ' + (r.error || 'could not open')
          + ' \\u2014 removed from recents';
        jset('ll-recents', jget('ll-recents', []).filter(x => x.path !== e.path));
        renderRecents();
      });
      const p = document.createElement('span');
      p.className = 'll-hint';
      p.style.marginTop = '0';
      p.textContent = e.path;
      row.appendChild(b);
      row.appendChild(p);
      box.appendChild(row);
    }
  }

  // ---- shift-drag on the heatmap adds an ignore range ----
  function initHeatmapIgnore() {
    const stack = document.getElementById('stack');
    const mapc = document.getElementById('map');
    if (!stack || !mapc || !dur) return;
    if (getComputedStyle(stack).position === 'static')
      stack.style.position = 'relative';
    let d0 = null, span = null, swallow = false;
    const fx = ev => {
      const r = mapc.getBoundingClientRect();
      return clamp01((ev.clientX - r.left) / r.width);
    };
    stack.addEventListener('click', ev => {
      if (swallow || ev.shiftKey) { ev.preventDefault(); ev.stopPropagation(); }
    }, true);
    stack.addEventListener('mousedown', ev => {
      if (!ev.shiftKey || ev.button !== 0) return;
      ev.preventDefault(); ev.stopPropagation();
      d0 = fx(ev);
      span = document.createElement('div');
      span.style.cssText = 'position:absolute;top:0;bottom:0;'
        + 'background:rgba(224,122,99,.22);border-left:1px solid #e07a63;'
        + 'border-right:1px solid #e07a63;pointer-events:none;z-index:6';
      stack.appendChild(span);
    }, true);
    stack.addEventListener('mousemove', ev => {
      if (d0 === null) return;
      ev.stopPropagation();
      const x = fx(ev), a = Math.min(d0, x), b = Math.max(d0, x);
      span.style.left = (a * 100) + '%';
      span.style.width = ((b - a) * 100) + '%';
    }, true);
    addEventListener('mouseup', ev => {
      if (d0 === null) return;
      const x = fx(ev), a = Math.min(d0, x) * dur, b = Math.max(d0, x) * dur;
      d0 = null;
      if (span) span.remove();
      span = null;
      swallow = true;  // the synthetic click that follows must not preview
      setTimeout(() => { swallow = false; }, 0);
      if (b - a < 0.05) return;
      const tok = a.toFixed(1) + '-' + b.toFixed(1);
      ignoreEl.value = ignoreEl.value.trim()
        ? ignoreEl.value.replace(/[,\\s]+$/, '') + ', ' + tok : tok;
      parseIgnore();
      saveVideoPrefs();
      setPanel(true);
      st.textContent = 'ignore ' + tok + 's added \\u2014 press Re-analyze to apply';
    }, true);
  }

  loadParams();
  fetch('/status').then(r => r.json()).then(s => {
    apply(s);
    loadVideoPrefs();
    renderRecents();
    if (s.phase === 'analyzing') poll();
    if (s.phase === 'loaded') setPanel(true);
    if (s.phase === 'ready') initHeatmapIgnore();
  });
})();
</script>
"""

LANDING = """<!doctype html><meta charset="utf-8"><title>looplab</title>
<body style="margin:0;background:#17151a;color:#ece7df;font:15px/1.6 system-ui,sans-serif">
__TOOLBAR__
<div style="max-width:640px;margin:64px auto;padding:0 24px">
  <h1 style="font-weight:600">Find the seamless loop in your footage</h1>
  <p style="color:#8f8798">Open a video of repetitive motion (a fidget toy, a pendulum,
  pouring, spinning). In the settings dropdown, optionally set an <b>attention
  crop</b> and time ranges to ignore, then press <b>Analyze</b>: looplab scores
  every possible cut pair and opens an interactive heatmap of the whole search
  space with previews and exports.</p>
  <p style="color:#8f8798">Everything runs locally; the file never leaves this
  machine.</p>
  <div id="ll-recents-wrap" hidden>
    <h2 style="font-weight:600;font-size:15px;margin:28px 0 4px">Recent videos</h2>
    <div id="ll-recents"></div>
  </div>
</div>
"""


def _toolbar() -> str:
    defaults = {k: getattr(core.SearchParams(), k)
                for k in ("min_loop", "max_loop", "window", "min_activity",
                          "focus_weight", "vel_weight")}
    defaults["proxy_long"] = "auto"
    return TOOLBAR.replace("__DEFAULTS__", json.dumps(defaults))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        with STATE.lock:
            d = str(STATE.workdir) if STATE.workdir else "."
        super().__init__(*a, directory=d, **kw)

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with STATE.lock:
                ready, wd = STATE.phase == "ready", STATE.workdir
            if ready and wd and (wd / "index.html").exists():
                html = (wd / "index.html").read_text()
                html = html.replace('<div class="wrap">',
                                    _toolbar() + '<div class="wrap">', 1)
                return self._html(html)
            return self._html(LANDING.replace("__TOOLBAR__", _toolbar()))
        if url.path == "/status":
            with STATE.lock:
                payload = {"phase": STATE.phase, "stage": STATE.stage,
                           "pct": STATE.pct, "video": STATE.video,
                           "info": STATE.info,
                           "result_info": STATE.result_info,
                           "error": STATE.error, "log": list(STATE.log)}
            payload["has_previous"] = bool(
                payload["video"]) and _has_previous(payload["video"])
            return self._json(payload)
        if url.path == "/frame":
            q = parse_qs(url.query)
            path = (q.get("path") or [""])[0]
            t = float((q.get("t") or ["1"])[0])
            if not path or not Path(path).exists():
                return self._json({"error": "file not found"}, 404)
            try:
                jpg = _poster_frame(path, t)
            except subprocess.CalledProcessError:
                return self._json({"error": "could not decode a frame"}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return
        with STATE.lock:
            has_wd = STATE.workdir is not None
        if not has_wd:  # static files exist only once results are loaded
            return self._json({"error": "not found"}, 404)
        return super().do_GET()

    def do_POST(self):
        if self.path == "/pick":
            try:
                path = _pick_file()
            except RuntimeError as e:
                return self._json({"path": None, "error": str(e)}, 500)
            if not path:
                return self._json({"path": None})
            try:
                _open_video(path)
            except Exception:
                return self._json(
                    {"path": None,
                     "error": "ffprobe could not read this file"}, 415)
            return self._json({"path": path,
                               "has_previous": _has_previous(path)})
        if self.path == "/open":
            path = self._body().get("path", "")
            if not path or not Path(path).exists():
                return self._json({"ok": False, "error": "file not found"}, 404)
            try:
                _open_video(path)
            except Exception:
                return self._json(
                    {"ok": False,
                     "error": "ffprobe could not read this file"}, 415)
            return self._json({"ok": True, "path": path,
                               "has_previous": _has_previous(path)})
        if self.path == "/load":
            data = self._body()
            with STATE.lock:
                path = data.get("path") or STATE.video or ""
            if not path or not Path(path).exists():
                return self._json({"ok": False, "error": "file not found"}, 404)
            if not _has_previous(path):
                return self._json(
                    {"ok": False,
                     "error": "no previous results for this video"}, 404)
            try:
                _adopt(path)
            except Exception:
                return self._json(
                    {"ok": False,
                     "error": "ffprobe could not read this file"}, 415)
            return self._json({"ok": True})
        if self.path == "/analyze":
            data = self._body()
            path = data.get("path", "")
            if not path or not Path(path).exists():
                return self._json({"ok": False, "error": "file not found"}, 404)
            with STATE.lock:
                busy = STATE.phase == "analyzing"
                known = STATE.video == path and STATE.info is not None
            if busy:
                return self._json({"ok": False, "error": "already analyzing"}, 409)
            if not known:
                try:
                    _open_video(path)
                except Exception:
                    return self._json(
                        {"ok": False,
                         "error": "ffprobe could not read this file"}, 415)
            threading.Thread(target=_analyze,
                             args=(path, data.get("params") or {}),
                             daemon=True).start()
            return self._json({"ok": True}, 202)
        if self.path == "/cancel":
            with STATE.lock:
                was = STATE.phase == "analyzing"
                STATE.cancel.set()
            return self._json({"ok": True, "was_analyzing": was})
        return self._json({"ok": False, "error": "unknown endpoint"}, 404)


def serve(port: int = 8321, initial: str | None = None,
          open_browser: bool = True) -> None:
    """Run the UI server (localhost only) until interrupted. The UI is live
    immediately and loads nothing on its own: `initial` (if given) is opened
    like a picked file - Analyze and, when a workdir exists, Load previous
    results are then one click away in the UI."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if initial:
        try:
            _open_video(str(Path(initial)))
            note = ("previous results available"
                    if _has_previous(initial) else "not yet analyzed")
            print(f"looplab: opened {Path(initial).name} ({note})",
                  file=sys.stderr)
        except Exception as e:
            print(f"looplab: could not open {initial}: {e}", file=sys.stderr)
    url = f"http://127.0.0.1:{port}/"
    print(f"looplab ui: {url}  (Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
