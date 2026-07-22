# looplab

Find the most seamless loop hidden in a video of repetitive motion — and cut it, frame-exact.

```bash
looplab input.mp4        # → input.loop.mp4, the best wrap point in the footage
```

![the looplab explorer: scrubbing the seam heatmap, browsing ranked candidates, previewing loops, zooming, exporting](docs/usage.gif)

Point it at footage of anything cyclic — a fidget toy, a pendulum, pouring, spinning, kneading — and looplab scores **every possible (start, end) cut pair** for how invisibly the video can wrap from its last frame back to its first, then renders the winner. The explorer above maps that entire search space: each pixel is one candidate loop (start time × loop length), and the bright ridges are the cuts that flow — hover, preview, and export any of them, not just the top pick.

And this is what comes out — the same 3-second cut played four times back to back, full quality. The wraps are at the 3/6/9-second marks, if you can find them:

https://github.com/user-attachments/assets/c12c0ffa-4e65-4973-851d-91a153600abf

## How it works

Frame similarity alone produces loops that *pose* correctly but *pop* on playback — the classic failure is matching two frames where the object is in the same place moving in opposite directions. looplab scores each candidate pair (s, e) with three distance streams, all computed on a small decoded proxy:

- **Position** — variance-weighted RGB distance, so moving regions outvote static background acreage.
- **Velocity** — central-difference frame derivatives, penalizing pose-matches with mismatched motion.
- **Focus** — luma masked to bright, low-saturation pixels: the moving prop in typical fidget footage. This is what catches "same hands, different toy configuration." Content-specific by design; `--focus-weight 0` disables it.

Each stream is evaluated over a **±K-frame window along the seam diagonal** (comparing s+k against e+k), so the motion must flow through the cut, not just match at it. Two gates then remove degenerate winners:

- **Activity** — a frozen or occluded stretch loops "perfectly" and shows nothing; loops must contain real motion relative to the video's median.
- **Disruption exclusion** — sustained framing anomalies (camera bumps, subject leaving frame) are auto-detected via MAD z-scores against the temporal median frame and excluded, while brief motion-blur spikes are kept as legitimate content. `--ignore A-B` adds manual exclusions on top for anything the detector shouldn't have to guess at.

The search is exhaustive over a **banded (start × length) space** — every start frame × every loop length within `[--min-loop, --max-loop]`. Cost is one video decode plus one banded distance computation, `O(N · B · d)` in frames × band width × proxy area: a minute of 30 fps video is ~150k scored pairs and runs in seconds. Longer footage scales past that quickly — a few minutes at a wide `--max-loop` is millions of pairs and a working set of tens of GB, so the proxy auto-steps down and `--proxy-long 256` is the lever when it drags.

**Backends** — picked automatically, forceable with `LOOPLAB_BACKEND=mlx|cupy|numpy`:

- **MLX** — Apple Silicon GPU via unified memory. Developed and tested here.
- **CUDA via CuPy** — NVIDIA GPUs (`pip install 'looplab[nvidia]'`). *Designed in but currently untested on real hardware*: CuPy shares the numpy code path verbatim (same array API), which is verified, but no CUDA device has run it yet — reports and fixes welcome. On ≤8 GB cards use `--proxy-long 256` to fit the working set.
- **numpy** — runs everywhere, same results, just slower.

## Install

Requires `ffmpeg`/`ffprobe` on PATH, Python ≥ 3.10.

```bash
pip install 'looplab[all] @ git+https://github.com/shihanqu/looplab'
# or minimal (numpy only, no explorer):
pip install 'looplab @ git+https://github.com/shihanqu/looplab'
```

Extras: `mlx` (Apple Silicon acceleration; skipped automatically on other platforms), `nvidia` (CuPy/CUDA 12 — untested, see Backends above), `explorer` (interactive heatmap UI, strips, heatmap.png).

## CLI

```bash
looplab input.mp4                     # best loop → input.loop.mp4
looplab input.mp4 -o perfect.mp4      # name the output
looplab input.mp4 --json              # machine-readable result on stdout
looplab input.mp4 --render-top 3      # also render ranks 2–3 into the workdir
looplab input.mp4 --explore           # + interactive heatmap explorer
```

Logs go to stderr; stdout carries only the output path (or JSON). Analysis artifacts — `scores.npz`, `candidates.json`, rendered extras — land in `<input>.looplab/`.

| Flag | Default | Meaning |
|---|---|---|
| `--min-loop` / `--max-loop` | 0.5 / 3.0 | loop length band, seconds |
| `--window` | 5 | ± frames matched across the seam |
| `--vel-weight` | 1.0 | velocity stream weight |
| `--focus-weight` | 1.0 | bright-object stream weight (0 = off) |
| `--min-activity` | 0.7 | min in-loop motion vs video median |
| `--proxy-long` | auto | proxy resolution (512/384/256 by length); lower = faster |
| `--crop` | full frame | attention crop: `X,Y,W,H` 0–1 rect the search looks at; output stays full-frame |
| `--ignore` | — | `A-B` seconds no loop may overlap (repeatable), e.g. `--ignore 42-47` |
| `--crf` | 18 | x264 quality of rendered loops |

## The explorer

```bash
looplab --ui                 # open the explorer, pick a video with the OS file dialog
looplab input.mp4 --ui       # same, with this video pre-opened
```

`--ui` starts a localhost-only server and opens the explorer shell immediately. Nothing runs or loads on its own: **Open video…** raises the native OS file picker (macOS `choose file`, tkinter elsewhere) — the server gets a real filesystem path and reads the original in place, no upload or copy — and **Analysis settings** then drives everything:

- **Tuning** — loop range, proxy resolution, seam window, gates, stream weights (persisted locally; every knob explains itself on hover).
- **Attention crop** — drag a rectangle on the live frame preview (drag inside it to move it); the seam *search* only looks inside it, rendered loops stay full-frame.
- **Ignore time ranges** — drag spans onto a filmstrip timeline whose hover scrubs the frame preview; click a span to remove it, overlapping spans merge, auto-detected disruptions show in gray, and a text field mirrors it all numerically. After a first pass you can also shift-drag a span directly on the heatmap.
- **Analyze / Re-analyze** — runs with a live weighted progress bar (decode → score → render) and a **Stop** button. When the video already has a `.looplab/` workdir, a **Load previous results** button appears instead of anything loading automatically — restarts are always explicit, and the landing page lists recent videos.

The explorer itself is a heatmap of the entire search space — loop start along x, loop length up y — so a bright vertical ridge is one moment in the footage that wraps well at several lengths. Hover to scrub any (start, end) pair with a magnetic cursor that snaps to ridge peaks, click any cell for an instant in-page segment preview, and one-click export the top cuts. Scroll over the map to zoom continuously about the cursor; **Fit** snaps back to the full-width default, which is also the zoom-out floor, and **1:1** jumps to one heatmap pixel per source frame. Clicking a ranked candidate scrolls the map to center it — that, plus zoom, is how you navigate a long video without hunting.

For a no-server snapshot, `--explore` writes the same UI as static files into the workdir: `index.html` (full-quality previews for the top 10) and `artifact.html` (single-file, videos embedded as data URIs — postable anywhere a strict CSP applies).

## For agents

looplab is built to be driven headless — see [SKILL.md](SKILL.md) for the full contract. Short version:

```bash
looplab input.mp4 --json --quiet
```

```json
{"ok": true, "output": "input.loop.mp4",
 "chosen": {"start_frame": 540, "end_frame": 608, "len_s": 2.265,
            "score": 0.489, "percentile": 0.0, "activity": 1.09},
 "candidates": ["… top 10, same shape …"]}
```

Exit codes: `0` success · `2` nothing survived the gates (relax `--min-activity`, widen the band) · `3` missing dependency · `4` unreadable input.

## Capture tips & limitations

Static camera, single continuous shot, locked exposure if possible. Perform **many repetitions**: each pair of cycles is another lottery ticket for the object landing in the same state twice, and candidate count grows quadratically with cycles. The focus stream assumes a bright, unsaturated subject; disable or reweight it for other footage. VFR phone video is handled by frame-index cutting with CFR re-stamping at the average rate. Apple **spatial video** (MV-HEVC, iPhone/Vision Pro capture) works via ffmpeg's multiview decode: the base (left-eye) view is analyzed and exports are flat H.264 — spatial re-encode is not attempted.

## License

MIT
