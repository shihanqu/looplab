"""Local UI server: pick a video with the OS file dialog, analyze, explore.

`looplab --ui` starts a localhost-only server and opens the explorer. The
page's "Open video..." button asks the *server* to raise the native OS file
dialog (macOS: osascript `choose file`; elsewhere: tkinter), so looplab gets a
real filesystem path and analyzes the original file in place — no upload, no
copy. Results are served from the video's `<name>.looplab/` workdir.
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

from . import core, explorer


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "idle"  # idle | analyzing | ready | error
        self.log: deque[str] = deque(maxlen=60)
        self.workdir: Path | None = None
        self.video: str | None = None
        self.error: str | None = None


STATE = _State()

TOOLBAR = """
<div id="ll-toolbar" style="position:sticky;top:0;z-index:50;display:flex;gap:14px;
  align-items:center;padding:10px 24px;background:#141217f2;border-bottom:1px solid #322e38;
  font:13px/1.4 system-ui,sans-serif;color:#ece7df;backdrop-filter:blur(4px)">
  <strong style="letter-spacing:.06em">looplab</strong>
  <button id="ll-open" style="background:#e07a63;border:none;border-radius:6px;color:#fff;
    padding:7px 14px;font:600 13px system-ui;cursor:pointer">Open video&hellip;</button>
  <span id="ll-status" style="color:#8f8798;font-family:ui-monospace,Menlo,monospace;
    font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
</div>
<script>
(function () {
  const btn = document.getElementById('ll-open');
  const st = document.getElementById('ll-status');
  async function poll() {
    const s = await (await fetch('/status')).json();
    if (s.phase === 'analyzing') {
      st.textContent = s.log[s.log.length - 1] || 'analyzing...';
      setTimeout(poll, 700);
    } else if (s.phase === 'ready') {
      st.textContent = 'ready - loading explorer...';
      location.reload();
    } else if (s.phase === 'error') {
      st.textContent = 'error: ' + (s.error || 'analysis failed');
      btn.disabled = false;
    }
  }
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    st.textContent = 'choose a file in the dialog...';
    try {
      const r = await (await fetch('/pick', { method: 'POST' })).json();
      if (!r.path) { st.textContent = ''; btn.disabled = false; return; }
      await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: r.path }),
      });
      st.textContent = 'analyzing ' + r.path.split('/').pop() + '...';
      poll();
    } catch (e) {
      st.textContent = 'error: ' + e;
      btn.disabled = false;
    }
  });
  fetch('/status').then(r => r.json()).then(s => {
    if (s.phase === 'analyzing') { btn.disabled = true; poll(); }
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
  <p style="color:#8f8798">Everything runs locally; the file never leaves this machine.</p>
</div>
"""


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
    except Exception as e:  # headless box, no tk — CLI still works
        raise RuntimeError(f"no native file dialog available ({e}); "
                           "run: looplab <video>  then  looplab --ui") from e


def _analyze(path: str) -> None:
    with STATE.lock:
        STATE.phase, STATE.video, STATE.error = "analyzing", path, None
        STATE.log.clear()

    def log(msg: str) -> None:
        with STATE.lock:
            STATE.log.append(msg)

    try:
        src = Path(path)
        result = core.run_search(str(src), core.SearchParams(), log=log)
        workdir = src.with_suffix(src.suffix + ".looplab")
        core.save_workdir(result, workdir)
        if not result.candidates:
            raise RuntimeError("no seam survived the gates - see SKILL.md tuning")
        log("[render] cutting candidate loops + scrub proxy (takes a minute)...")
        explorer.render_explorer_assets(result, workdir)
        explorer.build(result, workdir, mode="local")
        explorer.build(result, workdir, mode="artifact")
        with STATE.lock:
            STATE.workdir, STATE.phase = workdir, "ready"
    except Exception as e:
        with STATE.lock:
            STATE.phase, STATE.error = "error", str(e)


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
        if self.path.split("?")[0] in ("/", "/index.html"):
            with STATE.lock:
                ready, wd = STATE.phase == "ready", STATE.workdir
            if ready and wd and (wd / "index.html").exists():
                html = (wd / "index.html").read_text()
                html = html.replace('<div class="wrap">',
                                    TOOLBAR + '<div class="wrap">', 1)
                return self._html(html)
            return self._html(LANDING.replace("__TOOLBAR__", TOOLBAR))
        if self.path == "/status":
            with STATE.lock:
                return self._json({"phase": STATE.phase, "video": STATE.video,
                                   "error": STATE.error, "log": list(STATE.log)})
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
                path = json.loads(self.rfile.read(n) or b"{}").get("path", "")
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            if not path or not Path(path).exists():
                return self._json({"ok": False, "error": "file not found"}, 404)
            with STATE.lock:
                busy = STATE.phase == "analyzing"
            if busy:
                return self._json({"ok": False, "error": "already analyzing"}, 409)
            threading.Thread(target=_analyze, args=(path,), daemon=True).start()
            return self._json({"ok": True}, 202)
        return self._json({"ok": False, "error": "unknown endpoint"}, 404)


def serve(port: int = 8321, initial: str | None = None,
          open_browser: bool = True) -> None:
    """Run the UI server (localhost only) until interrupted."""
    if initial:
        threading.Thread(target=_analyze, args=(initial,), daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"looplab ui: {url}  (Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
