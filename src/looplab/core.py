"""Seam search: score every (start, end) frame pair in a banded search space.

A loop is a pair (s, e) where frame e can invisibly replace frame s: the cut
plays frames [s, e-1] and wraps. A pair scores well only when position,
velocity, and the bright-object state all match across a temporal window
straddling both endpoints, so motion flows through the seam instead of merely
posing at it.

Heavy math runs on MLX (Apple Silicon GPU via unified memory) when available
and falls back to numpy transparently.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BACKEND: tuple[str, object] | None = None


def get_backend() -> tuple[str, object]:
    """Pick the compute backend once: mlx (Apple Silicon) > cupy (NVIDIA CUDA)
    > numpy. Override with LOOPLAB_BACKEND=mlx|cupy|numpy — forcing an
    unavailable backend raises instead of silently falling back."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    forced = os.environ.get("LOOPLAB_BACKEND", "").strip().lower() or None
    if forced not in (None, "mlx", "cupy", "numpy"):
        raise RuntimeError(
            f"LOOPLAB_BACKEND must be mlx, cupy, or numpy (got {forced!r})")

    if forced in (None, "mlx"):
        try:
            import mlx.core as mx_mod
            _BACKEND = ("mlx", mx_mod)
            return _BACKEND
        except ImportError as e:
            if forced == "mlx":
                raise RuntimeError(
                    "LOOPLAB_BACKEND=mlx but mlx is not installed "
                    "(pip install 'looplab[mlx]', Apple Silicon only)") from e

    if forced in (None, "cupy"):
        cp_mod, count = None, 0
        try:
            import cupy as cp_mod
            count = cp_mod.cuda.runtime.getDeviceCount()
        except Exception as e:
            if forced == "cupy":
                raise RuntimeError(
                    f"LOOPLAB_BACKEND=cupy but CUDA is unavailable: {e}") from e
        if cp_mod is not None and count > 0:
            _BACKEND = ("cupy", cp_mod)
            return _BACKEND
        if forced == "cupy":
            raise RuntimeError("LOOPLAB_BACKEND=cupy but no CUDA device was found")

    _BACKEND = ("numpy", np)
    return _BACKEND


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class SearchParams:
    min_loop: float = 0.5          # seconds
    max_loop: float = 3.0          # seconds
    proxy_long: int | None = None  # proxy long side px; None = auto by length
    window: int = 5                # +/- frames matched across the seam
    sigma: float = 2.0             # gaussian width of that window
    vel_weight: float = 1.0        # velocity stream weight
    focus_weight: float = 1.0      # bright-object stream weight (0 disables)
    var_gamma: float = 1.0         # temporal-variance pixel weighting exponent
    min_activity: float = 0.7      # min in-loop motion vs video median
    top: int = 10                  # candidates to keep
    nms: int = 8                   # near-duplicate radius, frames
    crop: tuple[float, float, float, float] | None = None  # normalized x,y,w,h
                                   # attention crop: region the SEARCH looks at;
                                   # output stays full-frame
    ignore_ranges: list[tuple[float, float]] | None = None  # times (s) no loop
                                   # may overlap; merged into the disruption
                                   # exclusion mask


@dataclass
class SearchResult:
    src: str
    fps: float
    n_frames: int
    offsets: np.ndarray            # loop lengths (frames) scored, ascending
    s_win: np.ndarray              # windowed seam distance, (N, len(offsets))
    s_gated: np.ndarray            # same, NaN where gated
    activity: np.ndarray           # in-loop motion vs median, same shape
    exclude: np.ndarray            # per-frame disruption mask (bool, N)
    excluded_runs: list[tuple[int, int]]
    period_s: float
    quarter_periods_s: list[float]
    candidates: list[dict] = field(default_factory=list)
    params: SearchParams = field(default_factory=SearchParams)
    backend: str = "numpy"

    def to_manifest(self) -> dict:
        return {
            "src": self.src,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "period_s": round(self.period_s, 3),
            "quarter_periods_s": self.quarter_periods_s,
            "backend": self.backend,
            "excluded_runs": [
                {"first_frame": a, "last_frame": b,
                 "start_s": round(a / self.fps, 2), "end_s": round(b / self.fps, 2)}
                for a, b in self.excluded_runs
            ],
            "params": vars(self.params),
            "candidates": self.candidates,
        }


def ffprobe_video(src: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
         "-of", "json", src],
        capture_output=True, check=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    return {"width": int(st["width"]), "height": int(st["height"]),
            "fps": float(num) / float(den),
            "nb_frames": int(st.get("nb_frames", 0)),
            "duration": float(st.get("duration", 0.0))}


def _auto_proxy(n_frames_est: int) -> int:
    """Pick a proxy resolution the working set can afford as videos get long."""
    if n_frames_est <= 5000:
        return 512
    if n_frames_est <= 12000:
        return 384
    return 256


def decode_proxy(src: str, w: int, h: int,
                 crop: tuple[float, float, float, float] | None = None,
                 expected_frames: int = 0, progress=None) -> np.ndarray:
    """Decode the whole video once to a small RGB proxy, passthrough frame
    order. `crop` (normalized x,y,w,h) restricts the decoded region so the
    search only sees it. Streams stdout so progress can be reported."""
    vf = f"scale={w}:{h}"
    if crop:
        cx, cy, cw, ch = crop
        vf = (f"crop=trunc(iw*{cw:.6f}/2)*2:trunc(ih*{ch:.6f}/2)*2:"
              f"trunc(iw*{cx:.6f}):trunc(ih*{cy:.6f})," + vf)
    cmd = ["ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0", "-vsync", "0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    buf = bytearray()
    try:
        while True:
            b = proc.stdout.read(1 << 23)
            if not b:
                break
            buf += b
            if progress and expected_frames:
                progress(min(len(buf) / (frame_bytes * expected_frames), 1.0))
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=err)
    finally:  # progress may raise (cancellation) - never orphan the decoder
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    n = len(buf) // frame_bytes
    if n == 0 or len(buf) % frame_bytes:
        raise RuntimeError("decode produced no complete frames; check the input")
    return np.frombuffer(buf, np.uint8).reshape(n, h, w, 3)


def variance_weights(frames: np.ndarray, gamma: float) -> np.ndarray:
    """Per-pixel weight from temporal std, so moving regions drive the metric
    instead of acreage of static background. Mean-normalized, top-clipped so a
    few twinkly pixels can't dominate."""
    x = frames.reshape(frames.shape[0], -1).astype(np.float32)
    sd = x.std(axis=0)
    w = (sd / (sd.mean() + 1e-9)) ** gamma
    w = np.clip(w, 0.0, np.percentile(w, 99.5))
    return (w / w.mean()).astype(np.float32)


def _focus_stream_xp(frames: np.ndarray, xp):
    """Bright-object masked luma; xp is numpy or cupy (same array API)."""
    xr = xp.asarray(frames).astype(xp.float32) / 255.0
    y = xr[..., 0] * 0.299 + xr[..., 1] * 0.587 + xr[..., 2] * 0.114
    cmax = xr.max(axis=-1)
    cmin = xr.min(axis=-1)
    sat = (cmax - cmin) / (cmax + 1e-6)
    m = xp.clip((y - 0.55) / 0.20, 0.0, 1.0) * xp.clip((0.35 - sat) / 0.35, 0.0, 1.0)
    t = (y * m).reshape(frames.shape[0], -1).astype(xp.float32)
    return t, float((m > 0.5).mean())


def _focus_stream_mx(frames: np.ndarray, mx):
    n = frames.shape[0]
    xr = mx.array(frames).astype(mx.float32) * (1.0 / 255.0)
    y = xr[..., 0] * 0.299 + xr[..., 1] * 0.587 + xr[..., 2] * 0.114
    cmax = mx.max(xr, axis=-1)
    cmin = mx.min(xr, axis=-1)
    sat = (cmax - cmin) / (cmax + 1e-6)
    m = mx.clip((y - 0.55) / 0.20, 0.0, 1.0) * mx.clip((0.35 - sat) / 0.35, 0.0, 1.0)
    t = (y * m).reshape(n, -1)
    mx.eval(t)
    return t, float(mx.mean(m > 0.5))


def band_distances(frames: np.ndarray, offsets: np.ndarray, weights: np.ndarray,
                   use_focus: bool, log=log_stderr, progress=None):
    """Mean-squared distance of three streams for every (start, offset) cell:
    position (variance-weighted RGB), velocity (central difference), and the
    bright-object 'focus' stream (luma masked to bright, unsaturated pixels —
    the moving prop in a fidget video). NaN where start+offset runs off the end.

    Backends: MLX has its own path (different API); numpy and CuPy share one
    generic path, so the CUDA code is structurally identical to the tested
    CPU code.
    """
    backend, mod = get_backend()
    n = frames.shape[0]
    n_off = len(offsets)
    out = {k: np.full((n, n_off), np.nan, np.float32)
           for k in ("position", "velocity", "focus")}

    if backend == "mlx":
        mx = mod
        if progress:
            progress(0.0)  # cancellation checkpoint before heavy prep
        x = mx.array(frames).reshape(n, -1).astype(mx.float32) * (1.0 / 255.0)
        x = x - mx.mean(x, axis=1, keepdims=True)
        x = x * mx.array(np.sqrt(weights))
        v = mx.concatenate([x[1:2] - x[0:1], (x[2:] - x[:-2]) * 0.5,
                            x[-1:] - x[-2:-1]], axis=0)
        if progress:
            progress(0.0)
        t, cov = _focus_stream_mx(frames, mx) if use_focus else (None, 0.0)
        if use_focus:
            log(f"[focus] bright-object mask covers {cov * 100:.1f}% of pixels")
        for i, o in enumerate(offsets):
            o = int(o)
            ep = mx.mean(mx.square(x[o:] - x[:-o]), axis=1)
            ev = mx.mean(mx.square(v[o:] - v[:-o]), axis=1)
            evals = [ep, ev]
            if use_focus:
                et = mx.mean(mx.square(t[o:] - t[:-o]), axis=1)
                evals.append(et)
            mx.eval(*evals)
            out["position"][: n - o, i] = np.array(ep)
            out["velocity"][: n - o, i] = np.array(ev)
            if use_focus:
                out["focus"][: n - o, i] = np.array(et)
            if progress:
                progress((i + 1) / n_off)
        return out

    # numpy and cupy share this path; asnumpy is identity for numpy
    xp = mod
    if progress:
        progress(0.0)  # cancellation checkpoint before heavy prep
    x = xp.asarray(frames).reshape(n, -1).astype(xp.float32) / 255.0
    x -= x.mean(axis=1, keepdims=True)
    x *= xp.asarray(np.sqrt(weights))
    v = xp.concatenate([x[1:2] - x[0:1], (x[2:] - x[:-2]) * 0.5,
                        x[-1:] - x[-2:-1]], axis=0)
    if progress:
        progress(0.0)
    t, cov = _focus_stream_xp(frames, xp) if use_focus else (None, 0.0)
    if use_focus:
        log(f"[focus] bright-object mask covers {cov * 100:.1f}% of pixels")
    for i, o in enumerate(offsets):
        o = int(o)
        d = x[o:] - x[:-o]
        out["position"][: n - o, i] = asnumpy(xp.mean(d * d, axis=1))
        d = v[o:] - v[:-o]
        out["velocity"][: n - o, i] = asnumpy(xp.mean(d * d, axis=1))
        if use_focus:
            d = t[o:] - t[:-o]
            out["focus"][: n - o, i] = asnumpy(xp.mean(d * d, axis=1))
        if progress:
            progress((i + 1) / n_off)
    return out


def diagonal_window(d: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Smooth along the start axis per offset column == along the seam diagonal:
    at fixed offset, window term j compares (s+j, e+j). Cells whose window
    would leave the valid range become NaN."""
    taps = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (taps / sigma) ** 2).astype(np.float32)
    g /= g.sum()
    out = np.full_like(d, np.nan)
    for col in range(d.shape[1]):
        valid = np.count_nonzero(~np.isnan(d[:, col]))
        if valid <= 2 * k:
            continue
        c = np.convolve(d[:valid, col], g, mode="same")
        c[:k] = np.nan
        c[valid - k:] = np.nan
        out[:valid, col] = c
    return out


def motion_energy(frames: np.ndarray) -> np.ndarray:
    """Mean |frame difference| per transition t -> t+1, length N-1."""
    x = frames.reshape(frames.shape[0], -1).astype(np.float32)
    return np.abs(np.diff(x, axis=0)).mean(axis=1)


def activity_matrix(m: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean in-loop motion energy relative to the whole video's median, per
    (start, offset) cell. A frozen or occluded stretch scores ~0; typical ~1."""
    ref = np.median(m) + 1e-9
    p = np.concatenate([[0.0], np.cumsum(m)])
    n = len(m) + 1
    a = np.full((n, len(offsets)), np.nan, np.float32)
    for i, o in enumerate(offsets):
        o = int(o)
        a[: n - o, i] = (p[o - 1: n - 1] - p[: n - o]) / max(o - 1, 1) / ref
    return a


def estimate_tempo(m: np.ndarray, fps: float):
    """Dominant motion period via FFT autocorrelation of motion energy."""
    m = m - m.mean()
    n = len(m)
    ac = np.fft.irfft(np.abs(np.fft.rfft(m, 2 * n)) ** 2)[:n]
    ac /= ac[0] + 1e-12
    lo, hi = int(round(0.4 * fps)), min(int(round(4.0 * fps)), n - 1)
    period = (lo + int(np.argmax(ac[lo:hi]))) / fps

    quarters = []
    q = n // 4
    for qi in range(4):
        seg = m[qi * q:(qi + 1) * q]
        seg = seg - seg.mean()
        a = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * len(seg))) ** 2)[: len(seg)]
        h = min(hi, len(seg) - 1)
        quarters.append(round((lo + int(np.argmax(a[lo:h]))) / fps, 3)
                        if h > lo else float("nan"))
    return period, quarters


def anomaly_exclusion(frames: np.ndarray, fps: float, window_pad: int):
    """Frames belonging to sustained framing disruptions (camera excursions).

    Per-frame deviation from the per-pixel temporal median frame, MAD z-scored.
    Runs fire at z>3.5 with hysteresis down to z>2, nearby runs (<1s gap) merge,
    and only merged runs >=2s are excluded — brief blur or motion spikes are
    legitimate content and stay in."""
    x = frames.reshape(frames.shape[0], -1).astype(np.float32)
    med = np.median(x, axis=0)
    g = np.abs(x - med).mean(axis=1)
    z = (g - np.median(g)) / (np.median(np.abs(g - np.median(g))) + 1e-9)

    runs, i = [], 0
    while i < len(z):
        if z[i] > 3.5:
            j = i
            while j < len(z) and z[j] > 2.0:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < int(round(1.0 * fps)):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    exclude = np.zeros(len(z), bool)
    kept = []
    for a, b in merged:
        if (b - a) / fps >= 2.0:
            exclude[max(0, a - window_pad): min(len(z), b + window_pad)] = True
            kept.append((int(a), int(b)))
    return exclude, kept


def pick_candidates(s_gated: np.ndarray, activity: np.ndarray,
                    offsets: np.ndarray, fps: float, top: int, nms_frames: int):
    flat = s_gated.ravel()
    act = activity.ravel()
    order = np.argsort(flat)
    n_off = len(offsets)
    picked: list[dict] = []
    for idx in order:
        if np.isnan(flat[idx]):
            break
        s, oi = divmod(int(idx), n_off)
        e = s + int(offsets[oi])
        if any(abs(s - p["start_frame"]) <= nms_frames and
               abs(e - p["end_frame"]) <= nms_frames for p in picked):
            continue
        picked.append({
            "rank": len(picked) + 1,
            "start_frame": s, "end_frame": e,
            "start_s": round(s / fps, 3), "end_s": round(e / fps, 3),
            "len_frames": e - s, "len_s": round((e - s) / fps, 3),
            "score": float(flat[idx]),
            "activity": round(float(act[idx]), 3),
        })
        if len(picked) >= top:
            break
    valid = flat[~np.isnan(flat)]
    for p in picked:
        p["percentile"] = round(float((valid < p["score"]).mean() * 100.0), 4)
    return picked


def run_search(src: str, params: SearchParams | None = None,
               log=log_stderr, progress=None) -> SearchResult:
    """Full pipeline: probe -> proxy decode -> banded 3-stream scoring ->
    windowing -> gates -> ranked candidates.

    `progress(stage, frac)` gets stage in {"decode", "score", "post"} with
    frac in [0, 1] for live progress reporting."""
    params = params or SearchParams()
    t0 = time.time()

    info = ffprobe_video(src)
    fps = info["fps"]
    n_est = info["nb_frames"] or int(round(info["duration"] * fps))
    proxy_long = params.proxy_long or _auto_proxy(n_est)
    if params.proxy_long is None:
        log(f"[proxy] auto {proxy_long}px long side for ~{n_est} frames "
            f"(set --proxy-long to override)")
    crop = params.crop
    if crop:
        log(f"[crop] search restricted to x={crop[0]:.3f} y={crop[1]:.3f} "
            f"w={crop[2]:.3f} h={crop[3]:.3f} (output stays full-frame)")
    cw = info["width"] * (crop[2] if crop else 1.0)
    ch = info["height"] * (crop[3] if crop else 1.0)
    if cw >= ch:
        pw = proxy_long
        ph = max(16, int(round(pw * ch / cw / 2)) * 2)
    else:
        ph = proxy_long
        pw = max(16, int(round(ph * cw / ch / 2)) * 2)

    frames = decode_proxy(src, pw, ph, crop=crop, expected_frames=n_est,
                          progress=(lambda f: progress("decode", f))
                          if progress else None)
    n = frames.shape[0]
    log(f"[decode] {n} frames @ {fps:.3f} fps, proxy {pw}x{ph} "
        f"({time.time() - t0:.1f}s)")

    m = motion_energy(frames)
    period, quarters = estimate_tempo(m.copy(), fps)
    log(f"[tempo] dominant period {period:.2f}s (~{n / fps / period:.0f} cycles); "
        f"per-quarter {quarters}")

    offsets = np.arange(max(2, int(round(params.min_loop * fps))),
                        int(round(params.max_loop * fps)) + 1)
    weights = variance_weights(frames, params.var_gamma)
    use_focus = params.focus_weight > 0
    streams = band_distances(frames, offsets, weights, use_focus, log,
                             progress=(lambda f: progress("score", f))
                             if progress else None)

    d = streams["position"] / np.nanmedian(streams["position"])
    d += params.vel_weight * (streams["velocity"] / np.nanmedian(streams["velocity"]))
    if use_focus:
        d += params.focus_weight * (streams["focus"] / np.nanmedian(streams["focus"]))
    s_win = diagonal_window(d, params.window, params.sigma)

    activity = activity_matrix(m, offsets)
    exclude, bad_runs = anomaly_exclusion(frames, fps, params.window + 2)
    for a, b in bad_runs:
        log(f"[exclude] frames {a}..{b} ({a / fps:.2f}s..{b / fps:.2f}s) — "
            f"sustained framing disruption")
    if params.ignore_ranges:
        pad = params.window + 2
        for pair in params.ignore_ranges:  # careful: t0 above is the timer
            lo, hi = sorted(float(v) for v in pair)
            a = max(0, int(np.floor(lo * fps)) - pad)
            b = min(n, int(np.ceil(hi * fps)) + 1 + pad)
            if b > a:
                exclude[a:b] = True
                log(f"[ignore] frames {a}..{b - 1} ({lo:g}s..{hi:g}s) — "
                    f"user range")
    exc_cum = np.concatenate([[0], np.cumsum(exclude.astype(np.int32))])
    touches = np.zeros_like(s_win, bool)
    for i, o in enumerate(offsets):
        o = int(o)
        touches[: n - o, i] = (exc_cum[o + 1:] - exc_cum[: n - o]) > 0
    s_gated = np.where((activity >= params.min_activity) & ~touches, s_win, np.nan)

    n_valid = int(np.count_nonzero(~np.isnan(s_win)))
    n_kept = int(np.count_nonzero(~np.isnan(s_gated)))
    backend = get_backend()[0]
    log(f"[score] band {offsets[0]}..{offsets[-1]} frames "
        f"({offsets[0] / fps:.2f}..{offsets[-1] / fps:.2f}s), "
        f"{n_valid} pairs scored, {n_valid - n_kept} gated "
        f"({time.time() - t0:.1f}s, backend={backend})")

    cands = pick_candidates(s_gated, activity, offsets, fps,
                            params.top, params.nms)
    if progress:
        progress("post", 1.0)
    return SearchResult(
        src=str(src), fps=fps, n_frames=n, offsets=offsets,
        s_win=s_win, s_gated=s_gated, activity=activity,
        exclude=exclude, excluded_runs=bad_runs,
        period_s=period, quarter_periods_s=quarters,
        candidates=cands, params=params,
        backend=backend,
    )


def save_workdir(result: SearchResult, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        workdir / "scores.npz",
        s_win=result.s_win, s_gated=result.s_gated, activity=result.activity,
        exclude=result.exclude, offsets=result.offsets,
        fps=result.fps, n_frames=result.n_frames)
    (workdir / "candidates.json").write_text(
        json.dumps(result.to_manifest(), indent=2))
