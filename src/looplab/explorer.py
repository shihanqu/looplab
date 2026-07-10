"""Build the interactive heatmap explorer (HTML) from a SearchResult.

Two flavors from one template:
  local    — relative asset URLs; every candidate previews/exports the real cut
             (serve the output dir over HTTP, or open index.html directly)
  artifact — fully self-contained; scrub video and top-3 exports embedded as
             data URIs (safe for strict-CSP hosts like claude.ai artifacts)
"""

from __future__ import annotations

import base64
import io
import json
from importlib import resources
from pathlib import Path

import numpy as np

from . import render
from .core import SearchResult


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _quantize_grid(result: SearchResult) -> np.ndarray:
    q = -np.log(result.s_gated + 1e-9)
    vmin = np.nanpercentile(q, 60)
    vmax = np.nanmax(q)
    scaled = (q - vmin) / (vmax - vmin)
    return np.where(np.isnan(q), 255,
                    np.clip(np.round(scaled * 254), 0, 254)).astype(np.uint8)


def _strip_jpegs(out_dir: Path, ranks=(1, 2, 3), width: int = 1100) -> dict:
    from PIL import Image  # lazy: explorer extra

    strips = {}
    for r in ranks:
        p = out_dir / f"wrap_rank{r}.png"
        if p.exists():
            im = Image.open(p).convert("RGB")
            im = im.resize((width, int(im.height * width / im.width)),
                           Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=68)
            strips[str(r)] = base64.b64encode(buf.getvalue()).decode()
    return strips


def build(result: SearchResult, out_dir: Path, mode: str = "local",
          embed_exports: int = 3) -> Path:
    """Write index.html (local) or artifact.html (artifact) into out_dir.
    Expects loop cuts / scrub proxies already rendered there (see cli)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = result.candidates
    grid = _quantize_grid(result)

    assets: dict = {}
    for c in cands:
        s, e = c["start_frame"], c["end_frame"]
        name = f"loop_s{s}_e{e}.mp4"
        if mode == "local":
            if (out_dir / name).exists():
                assets[str(c["rank"])] = {"preview": name, "export": name,
                                          "name": name}
        elif c["rank"] <= embed_exports:
            ex = out_dir / f"export_s{s}_e{e}.mp4"
            if ex.exists():
                assets[str(c["rank"])] = {
                    "export": "data:video/mp4;base64," + _b64(ex), "name": name}

    if mode == "local":
        scrub = "scrub_480.mp4" if (out_dir / "scrub_480.mp4").exists() else None
    else:
        p = out_dir / "scrub_480_c30.mp4"
        scrub = "data:video/mp4;base64," + _b64(p) if p.exists() else None

    data = {
        "src": Path(result.src).name,
        "mode": mode,
        "n": result.n_frames,
        "n_off": int(len(result.offsets)),
        "off_min": int(result.offsets[0]),
        "fps": result.fps,
        "period_s": round(result.period_s, 3),
        "n_scored": int(np.count_nonzero(~np.isnan(result.s_gated))),
        "grid_b64": base64.b64encode(grid.ravel().tobytes()).decode(),
        "cands": cands,
        "strips": _strip_jpegs(out_dir),
        "assets": assets,
        "scrub": scrub,
    }

    html = (resources.files("looplab") / "artifact_template.html").read_text()
    html = html.replace("__DATA__", json.dumps(data))
    out = out_dir / ("index.html" if mode == "local" else "artifact.html")
    out.write_text(html)
    return out


def render_explorer_assets(result: SearchResult, out_dir: Path,
                           preview_ranks: int = 10) -> None:
    """Render everything the explorer links or embeds: full-quality cuts for
    the top candidates, embeddable exports for the top 3, scrub proxies."""
    fps = round(result.fps)
    for c in result.candidates[:preview_ranks]:
        s, e = c["start_frame"], c["end_frame"]
        p = out_dir / f"loop_s{s}_e{e}.mp4"
        if not p.exists():
            render.cut_loop(result.src, s, e, fps, p)
    for c in result.candidates[:3]:
        s, e = c["start_frame"], c["end_frame"]
        full = out_dir / f"loop_s{s}_e{e}.mp4"
        ex = out_dir / f"export_s{s}_e{e}.mp4"
        if not ex.exists():
            render.reencode(full, ex, crf=23)
    if not (out_dir / "scrub_480.mp4").exists():
        render.scrub_proxy(result.src, out_dir / "scrub_480.mp4", crf=28)
    if not (out_dir / "scrub_480_c30.mp4").exists():
        render.scrub_proxy(result.src, out_dir / "scrub_480_c30.mp4", crf=30)
    render.render_strips(result.src, result.candidates[:3], out_dir)


def render_heatmap_png(result: SearchResult, path: Path) -> Path:
    """Static heatmap (start x length, bright = better) with candidate rings."""
    import matplotlib  # lazy: explorer extra

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = -np.log(result.s_gated + 1e-9)
    vmin = np.nanpercentile(q, 60)
    vmax = np.nanmax(q)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#202020")
    fps = result.fps
    offsets = result.offsets
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    im = ax.imshow(q.T, origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=(0, q.shape[0] / fps, offsets[0] / fps,
                           offsets[-1] / fps))
    xs = [c["start_s"] for c in result.candidates]
    ys = [c["len_s"] for c in result.candidates]
    ax.scatter(xs, ys, s=90, facecolors="none", edgecolors="cyan",
               linewidths=1.2)
    for c in result.candidates[:5]:
        ax.annotate(str(c["rank"]), (c["start_s"], c["len_s"]),
                    textcoords="offset points", xytext=(6, 6),
                    color="cyan", fontsize=9, fontweight="bold")
    ax.set_xlabel("loop START (seconds into source) — END = start + length")
    ax.set_ylabel("loop LENGTH (s)")
    ax.set_title(f"{Path(result.src).name} — {result.n_frames} frames @ "
                 f"{fps:.2f}fps, period≈{result.period_s:.2f}s")
    fig.colorbar(im, ax=ax, pad=0.01, label="seam quality (bright = better)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
