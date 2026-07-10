"""ffmpeg render helpers: frame-exact loop cuts, encodes, seam strips."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


def cut_loop(src: str, start: int, end: int, fps: int, out: Path,
             crf: int = 18) -> Path:
    """Cut frames [start, end-1] from src and re-stamp as CFR. Frame `end`
    matches frame `start`, so it is excluded — playback wraps to `start`."""
    vf = f"select='between(n,{start},{end - 1})',setpts=N/{fps}/TB"
    _run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-an",
          "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
    return out


def reencode(src: Path, out: Path, crf: int, scale: str | None = None) -> Path:
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if scale:
        args += ["-vf", f"scale={scale}"]
    args += ["-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    _run(args)
    return out


def scrub_proxy(src: str, out: Path, crf: int = 28, width: int = 480) -> Path:
    """Whole-video downscaled proxy used by the explorer's segment previews."""
    _run(["ffmpeg", "-v", "error", "-y", "-i", src,
          "-vf", f"scale={width}:-2", "-an", "-c:v", "libx264",
          "-crf", str(crf), "-preset", "veryfast", "-g", "30",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
    return out


def render_strips(src: str, cands: list[dict], out_dir: Path,
                  panel_w: int = 270, panel_h: int = 480) -> None:
    """Seam frame strips for visual QA: one decode pass extracts every frame
    any strip needs; PIL composes. wrap_rankK.png shows played order across
    the wrap (E-3..E-1 | S..S+2); natural_rankK.png the true continuation."""
    from PIL import Image, ImageDraw  # lazy: explorer extra

    needed = sorted({f for c in cands
                     for f in list(range(c["end_frame"] - 3, c["end_frame"] + 3))
                     + list(range(c["start_frame"], c["start_frame"] + 3))})
    tmp = out_dir / "_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n,{f})" for f in needed)
    _run(["ffmpeg", "-v", "error", "-y", "-i", src,
          "-vf", f"select='{expr}',scale={panel_w}:{panel_h}",
          "-vsync", "0", "-frames:v", str(len(needed)),
          str(tmp / "f_%03d.png")])
    img_of = {f: Image.open(tmp / f"f_{i + 1:03d}.png")
              for i, f in enumerate(needed)}

    def strip(frame_ids, seam_after, title, path):
        w, h = panel_w * len(frame_ids), panel_h + 26
        canvas = Image.new("RGB", (w, h), (12, 12, 12))
        draw = ImageDraw.Draw(canvas)
        for i, f in enumerate(frame_ids):
            canvas.paste(img_of[f], (i * panel_w, 26))
            draw.text((i * panel_w + 6, 6), f"n={f}", fill=(230, 230, 230))
        draw.text((w - 220, 6), title, fill=(120, 200, 255))
        if seam_after is not None:
            x = seam_after * panel_w
            draw.rectangle([x - 2, 0, x + 2, h], fill=(0, 200, 255))
        canvas.save(path)

    for c in cands:
        s, e, r = c["start_frame"], c["end_frame"], c["rank"]
        strip([e - 3, e - 2, e - 1, s, s + 1, s + 2], 3, f"rank {r} WRAP",
              out_dir / f"wrap_rank{r}.png")
        strip([e - 3, e - 2, e - 1, e, e + 1, e + 2], 3, f"rank {r} NATURAL",
              out_dir / f"natural_rank{r}.png")
