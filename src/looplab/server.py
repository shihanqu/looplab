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
        return {"fps": m.get("fps"), "n_frames": m.get("n_frames"),
                "excluded_runs": [[r.get("start_s"), r.get("end_s")]
                                  for r in m.get("excluded_runs", [])]}
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
                STATE.result_info = {
                    "fps": result.fps, "n_frames": result.n_frames,
                    "excluded_runs": [[round(a / result.fps, 2),
                                       round(b / result.fps, 2)]
                                      for a, b in result.excluded_runs]}
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


def _poster_frame(path: str, t: float, w: int = 560) -> bytes:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(max(t, 0.0)), "-i", path,
         "-frames:v", "1", "-vf", f"scale={w}:-2", "-f", "image2pipe",
         "-vcodec", "mjpeg", "pipe:1"],
        capture_output=True, check=True)
    return r.stdout


TOOLBAR = """
<div id="ll-toolbar" style="position:sticky;top:0;z-index:50;background:#141217f2;
  border-bottom:1px solid #322e38;font:13px/1.4 system-ui,sans-serif;color:#ece7df;
  backdrop-filter:blur(4px)">
  <div style="display:flex;gap:12px;align-items:center;padding:9px 24px;flex-wrap:wrap">
    <strong style="letter-spacing:.06em;font-size:17px">looplab</strong>
    <button id="ll-open" class="ll-btn ll-accent">Open video&hellip;</button>
    <button id="ll-params" class="ll-btn ll-icon" title="Analysis settings"
      aria-expanded="false">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true">
        <line x1="3" y1="6" x2="21" y2="6"/><circle cx="9" cy="6" r="2.7"/>
        <line x1="3" y1="12" x2="21" y2="12"/><circle cx="15" cy="12" r="2.7"/>
        <line x1="3" y1="18" x2="21" y2="18"/><circle cx="7" cy="18" r="2.7"/>
      </svg>
      <span>Analysis settings</span>
    </button>
    <button id="ll-stop" class="ll-btn ll-stop" hidden
      title="Stop the analysis; partial results are discarded">Stop</button>
    <span id="ll-status" style="color:#8f8798;font-family:ui-monospace,Menlo,monospace;
      font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      max-width:52ch"></span>
    <a id="ll-gh" href="https://github.com/shihanqu/looplab" target="_blank"
      rel="noopener" title="Star looplab on GitHub">
      <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15
        -.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51
        -1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0
        0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2
        -.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29
        .25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16
        8c0-4.42-3.58-8-8-8Z"/>
      </svg>
      <span>Star</span>
      <span id="ll-ghn" hidden></span>
    </a>
  </div>
  <div id="ll-panel">
    <div class="ll-row">
      <label class="ll-f"><span class="ll-tt" data-tip="Shortest loop length considered, in seconds. Keep it at or above one full motion cycle.">min loop (s)</span><input id="p-min_loop" type="number" step="0.1" min="0.1"></label>
      <label class="ll-f"><span class="ll-tt" data-tip="Longest loop length considered. A wider band costs more compute; longer loops feel more organic.">max loop (s)</span><input id="p-max_loop" type="number" step="0.5" min="0.2"></label>
      <label class="ll-f"><span class="ll-tt" data-tip="Long side of the tiny analysis proxy. auto steps 512/384/256 down as videos get longer. Lower = faster but less discriminating.">proxy px</span><select id="p-proxy_long">
        <option value="auto">auto</option><option>512</option><option>384</option>
        <option>256</option><option>192</option></select></label>
      <label class="ll-f"><span class="ll-tt" data-tip="Frames matched on both sides of the cut (with Gaussian falloff) so motion flows through the seam instead of just posing at it.">seam window (&plusmn;f)</span><input id="p-window" type="number" step="1" min="1"></label>
      <label class="ll-f"><span class="ll-tt" data-tip="Minimum in-loop motion relative to the video median. Gates frozen or occluded stretches that would otherwise loop 'perfectly'. Lower it if everything gets gated.">min activity</span><input id="p-min_activity" type="number" step="0.1" min="0"></label>
      <label class="ll-f"><span class="ll-tt" data-tip="Weight of the bright-object stream: luma masked to bright, low-saturation pixels (the prop). Raise it if the prop state mismatches at the seam; set 0 for footage without a bright subject.">focus weight</span><input id="p-focus_weight" type="number" step="0.5" min="0"></label>
      <label class="ll-f"><span class="ll-tt" data-tip="Weight of the velocity stream, which penalizes frames that pose-match but move differently. Raise it if direction flips at the seam.">velocity weight</span><input id="p-vel_weight" type="number" step="0.5" min="0"></label>
    </div>
    <div class="ll-row" style="margin-bottom:8px">
      <div class="ll-sec">
        <span class="ll-sechead ll-tt" data-tip="Drag a rectangle on the frame preview below; drag inside the rectangle to move it. The seam search only scores pixels inside it - rendered loops stay full-frame.">attention crop</span>
        <div style="display:flex;gap:8px;align-items:center">
          <button id="ll-cropclear" class="ll-btn" title="Remove the attention crop" hidden>Clear</button>
          <span id="ll-cropval" class="ll-mono">full frame</span>
        </div>
      </div>
    </div>
    <div id="ll-source" hidden>
      <canvas id="ll-cropcv"></canvas>
      <canvas id="ll-tlcv" height="48"></canvas>
      <div id="ll-tllabel" class="ll-hint">&nbsp;</div>
    </div>
    <div class="ll-sec" style="margin:0 0 14px">
      <span class="ll-sechead ll-tt" data-tip="Time spans no loop may overlap. Drag on the timeline above to add one, click a span to remove it, or shift-drag the heatmap after an analysis. Gray spans are auto-detected disruptions.">ignore time ranges (s)</span>
      <input id="p-ignore" type="text" placeholder="e.g. 0-4.5, 42-47"
        spellcheck="false" autocomplete="off">
      <span class="ll-hint">drag the timeline to add &middot; click a span to remove &middot; overlaps merge</span>
    </div>
    <div class="ll-row" style="margin-bottom:0">
      <button id="ll-analyze" class="ll-btn ll-accent" disabled
        title="Run the seam search on the opened video with these settings">Analyze</button>
      <button id="ll-loadprev" class="ll-btn" hidden
        title="Serve this video's existing .looplab results without re-analyzing">Load previous results</button>
      <button id="ll-reset" class="ll-btn"
        title="Restore default tuning and clear this video's crop and ignore ranges">Reset</button>
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
  .ll-icon { display:inline-flex; align-items:center; gap:7px; padding:6px 11px; }
  #ll-gh { margin-left:auto; display:inline-flex; align-items:center; gap:6px;
    background:#2a2630; border:1px solid #4a4452; border-radius:6px; color:#ece7df;
    padding:5px 11px; font:600 12px system-ui; text-decoration:none; }
  #ll-gh:hover { border-color:#e07a63; }
  #ll-ghn { border-left:1px solid #4a4452; padding-left:9px; margin-left:2px;
    font:600 12px ui-monospace,Menlo,monospace; color:#d8d2c8; }
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
  #ll-panel.open { display:block; animation:llslide .16s ease-out;
    max-height:calc(100vh - 76px); overflow:auto; }
  #ll-source { margin:0 0 10px; }
  #ll-cropcv { display:block; margin:0 auto; border-radius:6px; cursor:crosshair;
    width:auto; height:auto; max-width:100%; max-height:46vh; }
  #ll-tlcv { display:block; width:100%; height:48px; margin-top:6px;
    border-radius:6px; cursor:crosshair; background:#1d1a21; }
  .ll-tt { border-bottom:1px dotted #6f6879; cursor:help; position:relative; }
  .ll-tt:hover::after { content:attr(data-tip); position:absolute; left:0;
    top:calc(100% + 6px); z-index:60; width:230px; background:#241f2a;
    color:#d8d2c9; border:1px solid #4a4452; border-radius:6px; padding:7px 10px;
    font:12px/1.45 system-ui,sans-serif; text-transform:none; letter-spacing:0;
    white-space:normal; pointer-events:none; box-shadow:0 4px 14px rgba(0,0,0,.4); }
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
  let llCrop = null, frameImg = null, autoRuns = [];
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
    renderTimeline();
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
  ignoreEl.addEventListener('change', () => {
    parseIgnore(); saveVideoPrefs(); renderTimeline();
  });

  function setPanel(open) {
    $('ll-panel').classList.toggle('open', open);
    $('ll-params').classList.toggle('on', open);
    $('ll-params').setAttribute('aria-expanded', String(open));
    if (open) renderTimeline();  // clientWidth is 0 while the panel is closed
  }
  $('ll-params').addEventListener('click',
    () => setPanel(!$('ll-panel').classList.contains('open')));

  // ---- source preview: crop drag on the frame, scrubbed by the timeline ----
  const cv = $('ll-cropcv'), tl = $('ll-tlcv'), tlLabel = $('ll-tllabel');
  let dragging = null;
  const fmtT = t => String(Math.round(t * 10) / 10);
  function frameURL(t, w) {
    return '/frame?path=' + encodeURIComponent(video)
      + '&t=' + t.toFixed(2) + '&w=' + w;
  }
  const frameCache = new Map();
  let frameSeq = 0;
  function showFrame(t) {
    if (!video) return;
    t = Math.min(Math.max(t, 0), Math.max((dur || 1) - 0.25, 0));
    const key = Math.round(t * 2) / 2;
    const seq = ++frameSeq;
    let img = frameCache.get(key);
    if (!img) {
      img = new Image();
      frameCache.set(key, img);
      img.src = frameURL(key, 560);
    }
    const use = () => {
      if (seq !== frameSeq || !img.naturalWidth) return;
      if (cv.width !== img.naturalWidth) {
        cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      }
      frameImg = img;
      drawPreview();
    };
    if (img.complete) use();
    else img.addEventListener('load', use, { once: true });
  }
  function updateCropUI() {
    $('ll-cropval').textContent = llCrop
      ? 'x=' + llCrop[0].toFixed(3) + ' y=' + llCrop[1].toFixed(3)
        + ' w=' + llCrop[2].toFixed(3) + ' h=' + llCrop[3].toFixed(3)
      : 'full frame';
    $('ll-cropclear').hidden = !llCrop;
    drawPreview();
  }
  function drawPreview() {
    if (!frameImg || !cv.width) return;
    const ctx = cv.getContext('2d');
    ctx.drawImage(frameImg, 0, 0, cv.width, cv.height);
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
  $('ll-cropclear').addEventListener('click', () => {
    llCrop = null; updateCropUI(); saveVideoPrefs();
  });
  const cvPos = ev => {
    const r = cv.getBoundingClientRect();
    return [clamp01((ev.clientX - r.left) / r.width),
            clamp01((ev.clientY - r.top) / r.height)];
  };
  const inCrop = p => llCrop && p[0] >= llCrop[0] && p[0] <= llCrop[0] + llCrop[2]
    && p[1] >= llCrop[1] && p[1] <= llCrop[1] + llCrop[3];
  cv.addEventListener('mousedown', ev => {
    const p = cvPos(ev);
    dragging = inCrop(p)
      ? { mode: 'move', ox: p[0] - llCrop[0], oy: p[1] - llCrop[1] }
      : { mode: 'draw', ax: p[0], ay: p[1] };
    ev.preventDefault();
  });
  cv.addEventListener('mousemove', ev => {  // hover affordance only
    if (!dragging) cv.style.cursor = inCrop(cvPos(ev)) ? 'move' : 'crosshair';
  });
  addEventListener('mousemove', ev => {  // active drags track past the canvas edge
    if (!dragging) return;
    const p = cvPos(ev);
    if (dragging.mode === 'draw')
      llCrop = [Math.min(dragging.ax, p[0]), Math.min(dragging.ay, p[1]),
                Math.abs(p[0] - dragging.ax), Math.abs(p[1] - dragging.ay)];
    else {
      llCrop[0] = Math.min(Math.max(p[0] - dragging.ox, 0), 1 - llCrop[2]);
      llCrop[1] = Math.min(Math.max(p[1] - dragging.oy, 0), 1 - llCrop[3]);
    }
    updateCropUI();
  });
  addEventListener('mouseup', () => {
    if (!dragging) return;
    const wasDraw = dragging.mode === 'draw';
    dragging = null;
    if (wasDraw && llCrop && (llCrop[2] < 0.02 || llCrop[3] < 0.02)) llCrop = null;
    updateCropUI();
    saveVideoPrefs();
  });

  // ---- timeline: hover scrubs the preview, drag adds an ignore range ----
  let tlHover = null, tlDrag = null, tlDown = null;
  const thumbs = [];
  let thumbsFor = null;
  function tlTime(ev) {
    const r = tl.getBoundingClientRect();
    if (!r.width) return 0;  // collapsed viewport: never emit NaN times
    return clamp01((ev.clientX - r.left) / r.width) * dur;
  }
  function writeRanges(list) {
    list.sort((p, q) => p[0] - q[0]);
    const merged = [];
    for (const r of list) {
      const last = merged[merged.length - 1];
      if (last && r[0] <= last[1] + 0.05) last[1] = Math.max(last[1], r[1]);
      else merged.push([r[0], r[1]]);
    }
    ignoreEl.value = merged.map(r => fmtT(r[0]) + '-' + fmtT(r[1])).join(', ');
    ignoreEl.classList.remove('ll-bad');
    saveVideoPrefs();
    renderTimeline();
  }
  function loadThumbs() {
    if (!video || !dur || thumbsFor === video) return;
    thumbsFor = video;
    thumbs.length = 0;
    const n = 12;
    for (let i = 0; i < n; i++) {
      const img = new Image();
      img.onload = () => renderTimeline();
      img.src = frameURL(Math.min(dur * (i + 0.5) / n, Math.max(dur - 0.25, 0)), 96);
      thumbs.push(img);
    }
  }
  function renderTimeline() {
    if (!video || !dur || !tl.clientWidth) return;
    const w = tl.clientWidth, h = tl.height;
    if (tl.width !== w) tl.width = w;
    const ctx = tl.getContext('2d');
    ctx.fillStyle = '#1d1a21';
    ctx.fillRect(0, 0, w, h);
    const n = thumbs.length || 12, tw = w / n;
    thumbs.forEach((img, i) => {
      if (!img.complete || !img.naturalWidth) return;
      const s = Math.max(tw / img.naturalWidth, h / img.naturalHeight);
      const dw = img.naturalWidth * s, dh = img.naturalHeight * s;
      ctx.save();
      ctx.beginPath(); ctx.rect(i * tw, 0, tw, h); ctx.clip();
      ctx.drawImage(img, i * tw + (tw - dw) / 2, (h - dh) / 2, dw, dh);
      ctx.restore();
    });
    const px = t => t / dur * w;
    ctx.fillStyle = 'rgba(120,118,130,.5)';
    for (const r of autoRuns)
      ctx.fillRect(px(r[0]), 0, Math.max(px(r[1]) - px(r[0]), 2), h);
    for (const r of (parseIgnore() || [])) {
      ctx.fillStyle = 'rgba(224,90,80,.45)';
      ctx.fillRect(px(r[0]), 0, Math.max(px(r[1]) - px(r[0]), 2), h);
      ctx.strokeStyle = '#e05a50';
      ctx.strokeRect(px(r[0]) + .5, .5, Math.max(px(r[1]) - px(r[0]), 2) - 1, h - 1);
    }
    if (tlDrag) {
      const a = Math.min(tlDrag[0], tlDrag[1]), b = Math.max(tlDrag[0], tlDrag[1]);
      ctx.fillStyle = 'rgba(224,122,99,.5)';
      ctx.fillRect(px(a), 0, Math.max(px(b) - px(a), 1), h);
    }
    if (tlHover !== null) {
      ctx.fillStyle = '#ece7df';
      ctx.fillRect(px(tlHover) - .5, 0, 1, h);
    }
  }
  tl.addEventListener('mousemove', ev => {
    if (!dur) return;
    const t = tlTime(ev);
    tlHover = t;
    if (tlDown !== null) tlDrag = [tlDown, t];
    const over = (parseIgnore() || []).find(r => t >= r[0] && t <= r[1]);
    tl.style.cursor = over && tlDown === null ? 'pointer' : 'crosshair';
    tlLabel.textContent = fmtT(t) + 's'
      + (over && tlDown === null
         ? ' \\u2014 click to remove ignore ' + fmtT(over[0]) + '-' + fmtT(over[1])
         : '');
    showFrame(t);
    renderTimeline();
  });
  tl.addEventListener('mouseleave', () => {
    tlHover = null;
    tlLabel.textContent = '\\u00a0';
    renderTimeline();
  });
  tl.addEventListener('mousedown', ev => {
    if (dur) { tlDown = tlTime(ev); ev.preventDefault(); }
  });
  addEventListener('mouseup', ev => {
    if (tlDown === null) return;
    const downT = tlDown, upT = tlTime(ev);
    tlDown = null; tlDrag = null;
    const a = Math.min(downT, upT), b = Math.max(downT, upT);
    const ranges = parseIgnore() || [];
    if (b - a < Math.max(dur * 0.004, 0.15)) {  // a click, not a drag
      const hit = ranges.findIndex(r => upT >= r[0] && upT <= r[1]);
      if (hit >= 0) writeRanges(ranges.filter((_, i) => i !== hit));
      else renderTimeline();
      return;
    }
    ranges.push([Math.max(0, a), Math.min(dur, b)]);
    writeRanges(ranges);
  });
  addEventListener('resize', () => renderTimeline());

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
    autoRuns = (s.result_info && s.result_info.excluded_runs) || [];
    $('ll-source').hidden = !video;
    if (video && dur && thumbsFor !== video) { showFrame(1); loadThumbs(); }
    renderTimeline();
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

  // live star count, cached an hour so the API is hit at most once per session
  (function stars() {
    const el = $('ll-ghn');
    if (!el) return;
    const show = n => { el.textContent = n; el.hidden = false; };
    const cached = jget('ll-ghstars', null);
    if (cached && Date.now() - cached.t < 3600e3) { show(cached.n); return; }
    fetch('https://api.github.com/repos/shihanqu/looplab')
      .then(r => r.ok ? r.json() : null)
      .then(j => {
        if (!j || typeof j.stargazers_count !== 'number') return;
        jset('ll-ghstars', { n: j.stargazers_count, t: Date.now() });
        show(j.stargazers_count);
      })
      .catch(() => {});
  })();

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
            try:
                t = float((q.get("t") or ["1"])[0])
                w = max(32, min(int(float((q.get("w") or ["560"])[0])), 1280))
            except ValueError:
                return self._json({"error": "bad t/w"}, 400)
            if not path or not Path(path).exists():
                return self._json({"error": "file not found"}, 404)
            try:
                jpg = _poster_frame(path, t, w)
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
