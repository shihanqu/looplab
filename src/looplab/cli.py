"""looplab CLI — agent-first: video in, seamless loop out.

    looplab input.mp4                 -> input.loop.mp4 (the #1 seam)
    looplab input.mp4 --json          -> machine-readable result on stdout
    looplab input.mp4 --explore       -> + interactive heatmap explorer

Logs go to stderr; stdout carries only the result (path or JSON).

Exit codes: 0 ok · 2 no candidate survived the gates · 3 missing dependency
(ffmpeg) · 4 unreadable input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, core, render


def _fail(args, code: int, msg: str) -> int:
    if args is not None and getattr(args, "json", False):
        print(json.dumps({"ok": False, "error": msg, "exit_code": code}))
    print(f"looplab: {msg}", file=sys.stderr)
    return code


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="looplab",
        description="Find and cut the most seamless loop in a video of "
                    "repetitive motion.")
    ap.add_argument("src", nargs="?",
                    help="input video (any ffmpeg-readable format); optional "
                         "with --ui, where it pre-loads the analysis")
    ap.add_argument("--ui", action="store_true",
                    help="start the local explorer UI: pick a video with the "
                         "native OS file dialog, analyze, browse the heatmap")
    ap.add_argument("--port", type=int, default=8321,
                    help="port for --ui (default 8321, localhost only)")
    ap.add_argument("-o", "--output",
                    help="output loop path (default: <src>.loop.mp4)")
    ap.add_argument("--json", action="store_true",
                    help="print a JSON result object on stdout")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress progress logs on stderr")
    ap.add_argument("--workdir",
                    help="analysis artifacts dir (default: <src>.looplab/)")
    ap.add_argument("--render-top", type=int, default=1, metavar="N",
                    help="render the top N loops (default 1; extras go to "
                         "the workdir)")
    ap.add_argument("--crf", type=int, default=18,
                    help="x264 CRF for rendered loops (default 18)")
    ap.add_argument("--explore", action="store_true",
                    help="build the interactive heatmap explorer + heatmap.png "
                         "(requires the 'explorer' extra)")

    tune = ap.add_argument_group("search tuning")
    tune.add_argument("--min-loop", type=float, default=0.5, metavar="S",
                      help="minimum loop length, seconds (default 0.5)")
    tune.add_argument("--max-loop", type=float, default=3.0, metavar="S",
                      help="maximum loop length, seconds (default 3.0)")
    tune.add_argument("--proxy-long", type=int, default=512, metavar="PX",
                      help="proxy long side; lower = faster (default 512)")
    tune.add_argument("--window", type=int, default=5, metavar="F",
                      help="+/- frames matched across the seam (default 5)")
    tune.add_argument("--vel-weight", type=float, default=1.0,
                      help="velocity-continuity stream weight (default 1)")
    tune.add_argument("--focus-weight", type=float, default=1.0,
                      help="bright-object stream weight; 0 disables "
                           "(default 1)")
    tune.add_argument("--min-activity", type=float, default=0.7,
                      help="min in-loop motion vs video median; gates frozen "
                           "or occluded stretches (default 0.7)")
    tune.add_argument("--top", type=int, default=10,
                      help="candidates to keep (default 10)")
    ap.add_argument("--version", action="version",
                    version=f"looplab {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return _fail(args, 3, "ffmpeg/ffprobe not found on PATH")
    if args.ui:
        from . import server
        server.serve(port=args.port, initial=args.src)
        return 0
    if not args.src:
        return _fail(args, 4, "missing input video (or pass --ui)")
    src = Path(args.src)
    if not src.exists():
        return _fail(args, 4, f"input not found: {src}")

    log = (lambda _msg: None) if args.quiet else core.log_stderr
    params = core.SearchParams(
        min_loop=args.min_loop, max_loop=args.max_loop,
        proxy_long=args.proxy_long, window=args.window,
        vel_weight=args.vel_weight, focus_weight=args.focus_weight,
        min_activity=args.min_activity, top=args.top)

    try:
        result = core.run_search(str(src), params, log=log)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode(errors="replace").strip()[-400:]
        return _fail(args, 4, f"ffmpeg could not read the input: {detail}")
    except RuntimeError as e:  # backend selection / configuration
        return _fail(args, 3, str(e))
    except Exception as e:
        if isinstance(e, MemoryError) or "OutOfMemory" in type(e).__name__:
            return _fail(args, 3,
                         "out of (GPU) memory — retry with --proxy-long 256 "
                         "or LOOPLAB_BACKEND=numpy")
        raise

    workdir = Path(args.workdir) if args.workdir else src.with_suffix(
        src.suffix + ".looplab")
    core.save_workdir(result, workdir)

    if not result.candidates:
        return _fail(args, 2,
                     "no seam survived the gates — try --min-activity 0.5, "
                     "a wider --max-loop, or check that the camera is static")

    fps = round(result.fps)
    out_path = Path(args.output) if args.output else src.with_suffix(".loop.mp4")
    rendered = []
    for c in result.candidates[: max(1, args.render_top)]:
        dst = out_path if c["rank"] == 1 else (
            workdir / f"loop_rank{c['rank']}_s{c['start_frame']}"
                      f"_e{c['end_frame']}.mp4")
        render.cut_loop(str(src), c["start_frame"], c["end_frame"], fps, dst,
                        crf=args.crf)
        rendered.append({"rank": c["rank"], "path": str(dst)})
        log(f"[render] rank {c['rank']} -> {dst}")

    explorer_path = None
    if args.explore:
        try:
            from . import explorer
            explorer.render_explorer_assets(result, workdir)
            explorer_path = explorer.build(result, workdir, mode="local")
            explorer.build(result, workdir, mode="artifact")
            explorer.render_heatmap_png(result, workdir / "heatmap.png")
            log(f"[explore] {explorer_path}")
        except ImportError:
            return _fail(args, 3,
                         "--explore needs the explorer extra: "
                         "pip install 'looplab[explorer]'")

    best = result.candidates[0]
    if args.json:
        print(json.dumps({
            "ok": True,
            "input": str(src),
            "output": str(out_path),
            "rendered": rendered,
            "workdir": str(workdir),
            "explorer": str(explorer_path) if explorer_path else None,
            "backend": result.backend,
            "fps": result.fps,
            "n_frames": result.n_frames,
            "period_s": round(result.period_s, 3),
            "chosen": best,
            "candidates": result.candidates,
        }, indent=None))
    else:
        print(out_path)
        log(f"[done] #1 seam {best['start_s']}s -> {best['end_s']}s "
            f"({best['len_s']}s, top {best['percentile']:.3f}% of scored pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
