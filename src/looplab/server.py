"""Local UI server: pick a video with the OS file dialog, analyze, explore.

`looplab --ui` starts a localhost-only server and opens the explorer shell
immediately - before any analysis. The "Open video..." button asks the
*server* to raise the native OS file dialog (macOS: osascript `choose file`;
elsewhere: tkinter), so looplab gets a real filesystem path and analyzes the
original file in place - no upload, no copy. Analysis streams a weighted
progress bar (decode -> score -> render); tuning parameters and an optional
search crop are set from the toolbar. Results are served from the video's
`<name>.looplab/` workdir.
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


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "idle"  # idle | analyzing | ready | error
        self.stage = ""
        self.pct = 0.0
        self.log: deque[str] = deque(maxlen=60)
        self.workdir: Path | None = None
        self.video: str | None = None
        self.error: str | None = None


STATE = _State()


def _progress(stage: str, frac: float) -> None:
    lo, hi = _WEIGHTS.get(stage, (0.0, 0.0))
    pct = (lo + (hi - lo) * min(max(frac, 0.0), 1.0)) * 100.0
    with STATE.lock:
        STATE.stage = stage
        STATE.pct = round(max(STATE.pct, pct), 1)  # monotonic


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
    return core.SearchParams(**kw)


def _analyze(path: str, raw_params: dict | None = None) -> None:
    with STATE.lock:
        STATE.phase, STATE.video, STATE.error = "analyzing", path, None
        STATE.stage, STATE.pct = "decode", 0.0
        STATE.log.clear()

    def log(msg: str) -> None:
        with STATE.lock:
            STATE.log.append(msg)

    try:
        params = _coerce_params(raw_params or {})
        src = Path(path)
        result = core.run_search(str(src), params, log=log, progress=_progress)
        workdir = src.with_suffix(src.suffix + ".looplab")
        core.save_workdir(result, workdir)
        if not result.candidates:
            raise RuntimeError("no seam survived the gates - lower min-activity "
                               "or widen the loop range in Params")
        log("[render] cutting candidate loops + scrub proxy...")
        explorer.render_explorer_assets(
            result, workdir,
            progress=lambda f: _progress("render", f))
        explorer.build(result, workdir, mode="local")
        explorer.build(result, workdir, mode="artifact")
        with STATE.lock:
            STATE.workdir, STATE.phase, STATE.pct = workdir, "ready", 100.0
    except Exception as e:
        with STATE.lock:
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
    <button id="ll-crop" class="ll-btn">Crop</button>
    <button id="ll-rerun" class="ll-btn" hidden>Re-analyze</button>
    <span id="ll-status" style="color:#8f8798;font-family:ui-monospace,Menlo,monospace;
      font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      max-width:46ch"></span>
  </div>
  <div id="ll-panel">
    <label class="ll-f">min loop (s)<input id="p-min_loop" type="number" step="0.1" min="0.1"></label>
    <label class="ll-f">max loop (s)<input id="p-max_loop" type="number" step="0.5" min="0.2"></label>
    <label class="ll-f">proxy px<select id="p-proxy_long">
      <option value="auto">auto</option><option>512</option><option>384</option>
      <option>256</option><option>192</option></select></label>
    <label class="ll-f">seam window (&plusmn;f)<input id="p-window" type="number" step="1" min="1"></label>
    <label class="ll-f">min activity<input id="p-min_activity" type="number" step="0.1" min="0"></label>
    <label class="ll-f">focus weight<input id="p-focus_weight" type="number" step="0.5" min="0"></label>
    <label class="ll-f">velocity weight<input id="p-vel_weight" type="number" step="0.5" min="0"></label>
    <button id="ll-reset" class="ll-btn">Reset</button>
  </div>
  <div id="ll-croppanel" hidden style="padding:10px 24px 14px;border-top:1px solid #2a2630">
    <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
      <div style="position:relative">
        <canvas id="ll-cropcv" style="display:block;border-radius:6px;cursor:crosshair;
          max-width:560px"></canvas>
      </div>
      <div style="font-size:12px;color:#8f8798;max-width:34ch">
        <p style="margin:0 0 8px">Drag a rectangle: the seam <em>search</em> only looks
        inside it. Rendered loops stay full-frame.</p>
        <p id="ll-cropval" style="font-family:ui-monospace,Menlo,monospace;margin:0 0 10px">
          crop: full frame</p>
        <button id="ll-cropclear" class="ll-btn">Clear crop</button>
      </div>
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
  .ll-f { display:flex; flex-direction:column; gap:4px; font-size:11px; color:#8f8798;
    text-transform:uppercase; letter-spacing:.08em; }
  .ll-f input, .ll-f select { width:96px; background:#1d1a21; color:#ece7df;
    border:1px solid #4a4452; border-radius:5px; padding:5px 7px; font:13px ui-monospace,Menlo,monospace; }
  .ll-icon { display:inline-flex; align-items:center; padding:6px 9px; }
  .ll-icon.on { background:#e07a63; border-color:#e07a63; color:#fff; }
  #ll-panel { display:none; padding:10px 24px 14px; border-top:1px solid #2a2630;
    gap:18px; flex-wrap:wrap; align-items:end; }
  #ll-panel.open { display:flex; animation:llslide .16s ease-out; }
  @keyframes llslide { from { opacity:0; transform:translateY(-5px); } }
  @media (prefers-reduced-motion: reduce) { #ll-panel.open { animation:none; } }
</style>
<script>
(function () {
  const $ = id => document.getElementById(id);
  const btn = $('ll-open'), st = $('ll-status'), bar = $('ll-bar'), rerun = $('ll-rerun');
  const P_KEYS = ['min_loop','max_loop','proxy_long','window','min_activity',
                  'focus_weight','vel_weight'];
  const DEFAULTS = __DEFAULTS__;
  let llCrop = null, lastVideo = null;

  function loadParams() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('ll-params') || '{}'); } catch (e) {}
    for (const k of P_KEYS) {
      const el = $('p-' + k);
      if (el) el.value = saved[k] ?? DEFAULTS[k];
    }
    if (Array.isArray(saved.crop)) { llCrop = saved.crop; updateCropLabel(); }
  }
  function gatherParams() {
    const out = {};
    for (const k of P_KEYS) { const el = $('p-' + k); if (el) out[k] = el.value; }
    if (llCrop) out.crop = llCrop;
    try { localStorage.setItem('ll-params', JSON.stringify(out)); } catch (e) {}
    return out;
  }
  $('ll-reset').addEventListener('click', () => {
    localStorage.removeItem('ll-params'); llCrop = null; loadParams(); updateCropLabel();
  });
  $('ll-params').addEventListener('click', () => {
    const open = $('ll-panel').classList.toggle('open');
    $('ll-params').classList.toggle('on', open);
    $('ll-params').setAttribute('aria-expanded', String(open));
    $('ll-croppanel').hidden = true;
  });

  // ---- crop picker ----
  const cv = $('ll-cropcv');
  let posterImg = null, dragging = null;
  function updateCropLabel() {
    $('ll-cropval').textContent = llCrop
      ? 'crop: x=' + llCrop[0].toFixed(3) + ' y=' + llCrop[1].toFixed(3)
        + ' w=' + llCrop[2].toFixed(3) + ' h=' + llCrop[3].toFixed(3)
      : 'crop: full frame';
  }
  function drawCrop() {
    if (!posterImg) return;
    const ctx = cv.getContext('2d');
    ctx.drawImage(posterImg, 0, 0, cv.width, cv.height);
    if (llCrop) {
      const [x, y, w, h] = llCrop;
      ctx.fillStyle = 'rgba(0,0,0,.55)';
      ctx.fillRect(0, 0, cv.width, y * cv.height);
      ctx.fillRect(0, (y + h) * cv.height, cv.width, cv.height);
      ctx.fillRect(0, y * cv.height, x * cv.width, h * cv.height);
      ctx.fillRect((x + w) * cv.width, y * cv.height, cv.width, h * cv.height);
      ctx.strokeStyle = '#e07a63'; ctx.lineWidth = 2;
      ctx.strokeRect(x * cv.width, y * cv.height, w * cv.width, h * cv.height);
    }
    updateCropLabel();
  }
  async function openCropPanel() {
    const s = await (await fetch('/status')).json();
    const video = lastVideo || s.video;
    if (!video) { st.textContent = 'open a video first, then set the crop'; return; }
    $('ll-croppanel').hidden = !$('ll-croppanel').hidden;
    $('ll-panel').classList.remove('open');
    $('ll-params').classList.remove('on');
    $('ll-params').setAttribute('aria-expanded', 'false');
    if ($('ll-croppanel').hidden || posterImg) { drawCrop(); return; }
    const img = new Image();
    img.onload = () => {
      posterImg = img;
      cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      drawCrop();
    };
    img.src = '/frame?path=' + encodeURIComponent(video) + '&t=1';
  }
  $('ll-crop').addEventListener('click', openCropPanel);
  $('ll-cropclear').addEventListener('click', () => { llCrop = null; drawCrop(); });
  const cvPos = ev => {
    const r = cv.getBoundingClientRect();
    return [Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1),
            Math.min(Math.max((ev.clientY - r.top) / r.height, 0), 1)];
  };
  cv.addEventListener('mousedown', ev => { dragging = cvPos(ev); });
  cv.addEventListener('mousemove', ev => {
    if (!dragging) return;
    const [x1, y1] = dragging, [x2, y2] = cvPos(ev);
    llCrop = [Math.min(x1, x2), Math.min(y1, y2),
              Math.abs(x2 - x1), Math.abs(y2 - y1)];
    drawCrop();
  });
  addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = null;
    if (llCrop && (llCrop[2] < 0.02 || llCrop[3] < 0.02)) { llCrop = null; drawCrop(); }
  });

  // ---- analyze + progress ----
  const STAGE_LABEL = { decode: 'decoding', score: 'scoring seams',
                        post: 'gating', render: 'rendering previews' };
  async function poll() {
    const s = await (await fetch('/status')).json();
    if (s.phase === 'analyzing') {
      bar.style.width = s.pct + '%';
      st.textContent = (STAGE_LABEL[s.stage] || s.stage) + ' ' + s.pct + '%'
        + (s.log.length ? ' - ' + s.log[s.log.length - 1] : '');
      setTimeout(poll, 600);
    } else if (s.phase === 'ready') {
      bar.style.width = '100%';
      st.textContent = 'ready - loading explorer...';
      location.reload();
    } else if (s.phase === 'error') {
      st.textContent = 'error: ' + (s.error || 'analysis failed');
      btn.disabled = false; rerun.disabled = false;
    }
  }
  async function analyze(path) {
    btn.disabled = true; rerun.disabled = true;
    bar.style.width = '0%';
    await fetch('/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, params: gatherParams() }),
    });
    st.textContent = 'analyzing ' + path.split('/').pop() + '...';
    poll();
  }
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    st.textContent = 'choose a file in the dialog...';
    try {
      const r = await (await fetch('/pick', { method: 'POST' })).json();
      if (!r.path) { st.textContent = ''; btn.disabled = false; return; }
      lastVideo = r.path; posterImg = null;
      analyze(r.path);
    } catch (e) { st.textContent = 'error: ' + e; btn.disabled = false; }
  });
  rerun.addEventListener('click', async () => {
    const s = await (await fetch('/status')).json();
    const video = lastVideo || s.video;
    if (video) analyze(video);
  });

  loadParams();
  fetch('/status').then(r => r.json()).then(s => {
    if (s.phase === 'analyzing') { btn.disabled = true; poll(); }
    if (s.video) { lastVideo = s.video; rerun.hidden = false; }
  });
})();
</script>
"""

LANDING = """<!doctype html><meta charset="utf-8"><title>looplab</title>
<body style="margin:0;background:#17151a;color:#ece7df;font:15px/1.6 system-ui,sans-serif">
__TOOLBAR__
<div style="max-width:640px;margin:80px auto;padding:0 24px">
  <h1 style="font-weight:600">Find the seamless loop in your footage</h1>
  <p style="color:#8f8798">Open a video of repetitive motion (a fidget toy, a pendulum,
  pouring, spinning). looplab scores every possible cut pair, then opens an
  interactive heatmap of the whole search space with previews and exports.</p>
  <p style="color:#8f8798">Tune the search with the sliders button, restrict it
  to a region with <b>Crop</b>. Everything runs locally; the file never leaves
  this machine.</p>
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
                return self._json({
                    "phase": STATE.phase, "stage": STATE.stage,
                    "pct": STATE.pct, "video": STATE.video,
                    "error": STATE.error, "log": list(STATE.log)})
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
        return super().do_GET()

    def do_POST(self):
        if self.path == "/pick":
            try:
                return self._json({"path": _pick_file()})
            except RuntimeError as e:
                return self._json({"path": None, "error": str(e)}, 500)
        if self.path == "/analyze":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            path = data.get("path", "")
            if not path or not Path(path).exists():
                return self._json({"ok": False, "error": "file not found"}, 404)
            with STATE.lock:
                busy = STATE.phase == "analyzing"
            if busy:
                return self._json({"ok": False, "error": "already analyzing"}, 409)
            threading.Thread(target=_analyze,
                             args=(path, data.get("params") or {}),
                             daemon=True).start()
            return self._json({"ok": True}, 202)
        return self._json({"ok": False, "error": "unknown endpoint"}, 404)


def serve(port: int = 8321, initial: str | None = None,
          open_browser: bool = True) -> None:
    """Run the UI server (localhost only) until interrupted. The UI is live
    immediately; analysis of `initial` (if given) runs in the background."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if initial:
        threading.Thread(target=_analyze, args=(initial, {}),
                         daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"looplab ui: {url}  (Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
