---
name: looplab
description: Find and cut the most seamless loop in a video of repetitive motion (fidget toys, pendulums, spinners, pouring). Use when the user wants a perfect/seamless/infinite loop, cinemagraph-style segment, or "GIF that never pops" from real footage. Headless CLI; outputs the best-seam MP4 plus ranked JSON candidates.
---

# looplab — seamless-loop extraction for agents

## When to use

The user has a video of a **repetitive motion from a static camera** and wants a segment that loops invisibly. Do not use for multi-shot edits, moving-camera footage, or synthetic loop *generation* — looplab finds cuts that already exist in the footage; it does not fabricate frames.

## Requirements

`ffmpeg` + `ffprobe` on PATH; Python ≥ 3.10. Install once:

```bash
pip install 'looplab[all] @ git+https://github.com/shihanqu/looplab'
```

Backends auto-select: MLX on Apple Silicon, CuPy on NVIDIA (`pip install 'looplab[nvidia]'`; designed in but untested on real CUDA hardware), numpy elsewhere — slower, same results. Force one with `LOOPLAB_BACKEND=mlx|cupy|numpy`; the JSON reports which ran in `backend`. GPU out-of-memory exits 3 with a hint — retry with `--proxy-long 256` or `LOOPLAB_BACKEND=numpy`.

## Canonical invocation

```bash
looplab INPUT.mp4 --json --quiet
```

- stdout: one JSON object (below). stderr: progress logs (suppressed by `--quiet`).
- Renders the #1 loop to `INPUT.loop.mp4` (override with `-o`).
- Writes `scores.npz` + `candidates.json` + extra renders to `INPUT.mp4.looplab/` (override with `--workdir`).

## JSON contract

```json
{
  "ok": true,
  "output": "clip.loop.mp4",
  "rendered": [{"rank": 1, "path": "clip.loop.mp4"}],
  "workdir": "clip.mp4.looplab",
  "backend": "mlx",
  "fps": 30.018, "n_frames": 2021, "period_s": 0.4,
  "chosen": {
    "rank": 1, "start_frame": 540, "end_frame": 608,
    "start_s": 17.989, "end_s": 20.254, "len_frames": 68, "len_s": 2.265,
    "score": 0.489, "activity": 1.09, "percentile": 0.0
  },
  "candidates": ["… up to --top entries, same shape, best first …"]
}
```

Semantics: the loop plays frames `[start_frame, end_frame - 1]` and wraps — frame `end_frame` visually matches frame `start_frame`. `score` is a relative distance (lower = better; only comparable within one run). `percentile` is the fraction of all scored pairs that beat this one (0.0 = the best seam in the video). `activity` is in-loop motion vs the video median (≈1 = typical motion; near the `--min-activity` floor = sluggish stretch, eyeball it).

On failure: `{"ok": false, "error": "...", "exit_code": N}` on stdout, same code as process exit. `0` ok · `2` no candidate survived the gates · `3` missing dependency · `4` unreadable input.

## Interpreting results / next actions

- **Trust `chosen` when** `percentile` ≤ ~0.05 and `activity` ≥ ~0.8. Deliver `output` directly.
- **Verify visually** when the top scores cluster tightly: render more with `--render-top 3` and compare, or build the human-facing explorer with `--explore` (adds `index.html` + self-contained `artifact.html` to the workdir).
- **Exit 2 (all gated)**: retry with `--min-activity 0.5`; if still failing, widen `--max-loop` (e.g. 5) or lower `--min-loop`. Persistent failure usually means camera motion or no true repetition.
- **Seam pops on a specific object** despite a good score: raise `--focus-weight` (2.0) — it weights the bright-object state stream. Footage without a bright subject: `--focus-weight 0`.
- **Slow machine / long video**: `--proxy-long 256` quarters the math at slight discrimination cost.

## Tuning quick reference

| Flag | Default | Raise when… | Lower when… |
|---|---|---|---|
| `--max-loop` | 3.0 | user wants longer, more organic loops | — |
| `--min-activity` | 0.7 | degenerate low-motion loops win | exit 2 / slow deliberate motion |
| `--focus-weight` | 1.0 | prop state mismatches at the seam | subject isn't bright/white |
| `--vel-weight` | 1.0 | direction flips at the seam | motion blur dominates |
| `--window` | 5 | seams pose-match but don't flow | very fast cycles (< 0.5 s) |

## Library use

```python
from looplab import run_search, SearchParams
result = run_search("clip.mp4", SearchParams(max_loop=4.0))
result.candidates[0]          # same dict shape as the JSON contract
```
