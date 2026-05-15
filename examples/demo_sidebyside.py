"""Side-by-side playback of the three benchmark pipelines for an on-site demo.

Each column replays the pre-rendered ``result_*.jpg`` visualizations produced by
one of the offline benchmark runs:

  1. ``hf_benchmark.py``                  -> Original (Baseline)
  2. ``model_encryption.py``              -> Encrypted (Unauthorized, wrong predictions)
  3. ``secure_inference_benchmark.py``    -> Secure Inference (Authorized, correct but slower)

Each column advances on its own timer derived from the real measured FPS of
that pipeline, multiplied by ``--slowdown`` so viewers can actually read the
boxes. The third column is therefore always slower than the first (its real
FPS is lower), making the throughput cost of decrypt-on-the-fly visible
without speeding past the wrong predictions in the middle column.

Example:

    python3 demo_sidebyside.py \
        --baseline_dir ./benchmark_results \
        --encrypted_dir ./secure_benchmark_results \
        --secure_dir   ./secure_inference_benchmark_results \
        --baseline_fps 10.69 --encrypted_fps 10.69 --secure_fps 7.41 \
        --slowdown 4.0
"""

import argparse
import glob
import json
import os
import sys
import time

import matplotlib


def _has_display() -> bool:
    """Return True if a GUI display is plausibly available.

    matplotlib.use('TkAgg') succeeds even on headless boxes because the Python
    module imports fine; the failure only surfaces later when Tk tries to open
    the display. Checking env vars up front lets us fall back cleanly.
    """
    if sys.platform == "darwin":
        return True
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _select_interactive_backend():
    """Pick a GUI matplotlib backend before pyplot is imported.

    The default in fresh venvs is often ``agg`` (non-interactive); with that
    backend ``plt.show()`` returns immediately and ``FuncAnimation`` gets
    garbage-collected without ever rendering.
    """
    if not _has_display():
        return None
    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "MacOSX"):
        try:
            matplotlib.use(candidate, force=True)
            return candidate
        except (ImportError, ValueError):
            continue
    return None


_INTERACTIVE_BACKEND = _select_interactive_backend()


def _wire_imageio_ffmpeg():
    """Point matplotlib at the ffmpeg binary bundled in the imageio-ffmpeg wheel.

    Lets users install ffmpeg with ``uv pip install imageio-ffmpeg`` instead of
    needing a system / conda binary on PATH. No-op if the package is not
    installed.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    matplotlib.rcParams["animation.ffmpeg_path"] = exe
    return exe


_IMAGEIO_FFMPEG = _wire_imageio_ffmpeg()

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402


def discover_images(folder: str):
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")
    paths = sorted(glob.glob(os.path.join(folder, "result_*.jpg")))
    if not paths:
        raise SystemExit(f"No result_*.jpg images in {folder}")
    return paths


def read_fps_from_dir(folder: str):
    """Return the measured FPS from any *.json results file in ``folder``.

    Looks for the ``performance_metrics.fps`` field written by hf_benchmark.py,
    model_encryption.py, and secure_inference_benchmark.py. Returns ``None`` if
    no JSON is found or the field is missing.
    """
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        perf = data.get("performance_metrics") if isinstance(data, dict) else None
        if isinstance(perf, dict) and isinstance(perf.get("fps"), (int, float)):
            return float(perf["fps"]), path
    return None, None


def preload(paths):
    return [mpimg.imread(p) for p in paths]


def main():
    parser = argparse.ArgumentParser(
        description="Three-column live demo of baseline vs encrypted vs secure inference."
    )
    parser.add_argument("--baseline_dir", default="./benchmark_results",
                        help="Output directory of hf_benchmark.py")
    parser.add_argument("--encrypted_dir", default="./secure_benchmark_results",
                        help="Output directory of model_encryption.py")
    parser.add_argument("--secure_dir", default="./secure_inference_benchmark_results",
                        help="Output directory of secure_inference_benchmark.py")
    parser.add_argument("--baseline_fps", type=float, default=None,
                        help="Override the FPS read from <baseline_dir>/*.json (performance_metrics.fps)")
    parser.add_argument("--encrypted_fps", type=float, default=None,
                        help="Override the FPS read from <encrypted_dir>/*.json (performance_metrics.fps)")
    parser.add_argument("--secure_fps", type=float, default=None,
                        help="Override the FPS read from <secure_dir>/*.json (performance_metrics.fps)")
    parser.add_argument("--fallback_fps", type=float, default=10.0,
                        help="FPS used when a directory has no readable JSON and no override was given")
    parser.add_argument("--slowdown", type=float, default=4.0,
                        help="Display each frame for slowdown / fps seconds so viewers can read")
    parser.add_argument("--figsize", type=float, nargs=2, default=(18.0, 7.0),
                        help="Matplotlib figure size in inches")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="(default) Loop each column forever once it reaches the last image")
    parser.add_argument("--save", type=str, default=None,
                        help="If set, render the animation to this file (e.g. demo.mp4 or demo.gif) "
                             "instead of opening a window. Useful on headless machines.")
    parser.add_argument("--save_seconds", type=float, default=30.0,
                        help="Duration to record when --save is used")
    parser.add_argument("--title_fontsize", type=float, default=20.0,
                        help="Font size for per-column titles")
    parser.add_argument("--suptitle_fontsize", type=float, default=16.0,
                        help="Font size for the figure-level subtitle")
    args = parser.parse_args()

    if args.save is None and _INTERACTIVE_BACKEND is None:
        print(
            "No GUI display detected (DISPLAY / WAYLAND_DISPLAY are unset).\n"
            "Either:\n"
            "  - re-run inside an X/Wayland session (or with `ssh -X`), or\n"
            "  - re-run with `--save demo.mp4` (or `--save demo.gif`) to render headlessly.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"matplotlib backend: {matplotlib.get_backend()}")

    folders = [args.baseline_dir, args.encrypted_dir, args.secure_dir]
    titles = [
        "Original (Baseline)",
        "Encrypted (Unauthorized)",
        "Secure Inference (Authorized)",
    ]
    overrides = [args.baseline_fps, args.encrypted_fps, args.secure_fps]

    real_fps = []
    for title, folder, override in zip(titles, folders, overrides):
        if override is not None:
            real_fps.append(override)
            print(f"  {title}: FPS={override:.2f} (CLI override)")
            continue
        fps, src = read_fps_from_dir(folder)
        if fps is None:
            print(f"  {title}: no FPS JSON found in {folder}; using fallback {args.fallback_fps:.2f}",
                  file=sys.stderr)
            real_fps.append(args.fallback_fps)
        else:
            print(f"  {title}: FPS={fps:.2f} (from {os.path.basename(src)})")
            real_fps.append(fps)
    intervals = [args.slowdown / f for f in real_fps]

    # Visual alignment: the encrypted run measures slightly differently from the
    # baseline (no decryption overhead, but small JIT / dataloader noise), which
    # makes the first two columns drift apart on screen even though they should
    # tell the same story about throughput. Force the encrypted column's
    # *display* interval to match the baseline's so they advance in lockstep.
    # Only the displayed pacing is changed; real_fps[1] still reflects the
    # measured value reported in the per-column title.
    intervals[1] = intervals[0]

    print("Loading visualizations...")
    sequences = []
    for folder in folders:
        paths = discover_images(folder)
        print(f"  {folder}: {len(paths)} frames")
        sequences.append(preload(paths))

    n_frames = [len(s) for s in sequences]

    fig, axes = plt.subplots(1, 3, figsize=tuple(args.figsize))
    try:
        fig.canvas.manager.set_window_title("ISCAS 2026 - Secure Inference Demo")
    except Exception:
        pass

    images = []
    indices = [0, 0, 0]
    next_advance = [time.monotonic() + iv for iv in intervals]
    for ax, title, seq, fps, iv in zip(axes, titles, sequences, real_fps, intervals):
        ax.set_title(
            f"{title}\nReal: {fps:.2f} FPS   |   Display: {1.0 / iv:.2f} FPS",
            fontsize=args.title_fontsize,
            fontweight="bold",
        )
        ax.axis("off")
        images.append(ax.imshow(seq[0]))

    fig.suptitle(
        f"Slowdown x{args.slowdown:g} (display = slowdown / real-FPS)",
        fontsize=args.suptitle_fontsize,
    )

    def tick(_frame):
        now = time.monotonic()
        for i in range(3):
            if now >= next_advance[i]:
                indices[i] = (indices[i] + 1) % n_frames[i]
                images[i].set_data(sequences[i][indices[i]])
                next_advance[i] = now + intervals[i]
        return images

    # 50 ms wall-clock tick — fine-grained enough to honor each column's interval
    # without burning CPU. blit=False because suptitle / titles do not blit cleanly.
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext == ".gif":
            writer = "pillow"
        else:
            from matplotlib.animation import FFMpegWriter
            if not FFMpegWriter.isAvailable():
                print(
                    f"Cannot write {args.save}: matplotlib's FFMpegWriter is unavailable.\n"
                    "Matplotlib needs the ffmpeg command-line tool (the PyPI `ffmpeg`\n"
                    "package is a different, unrelated wrapper). Install one of:\n"
                    "    uv pip install imageio-ffmpeg     # bundles a static ffmpeg in the venv\n"
                    "    sudo apt install ffmpeg           # system-wide\n"
                    "    conda install -c conda-forge -n base ffmpeg\n"
                    "Or rerun with `--save demo.gif` to use the pure-Python Pillow writer.",
                    file=sys.stderr,
                )
                sys.exit(3)
            writer = "ffmpeg"
            if _IMAGEIO_FFMPEG:
                print(f"Using ffmpeg from imageio-ffmpeg: {_IMAGEIO_FFMPEG}")
        tick_ms = 50
        n_save_frames = max(1, int(args.save_seconds * 1000 / tick_ms))
        anim = FuncAnimation(fig, tick, frames=n_save_frames, interval=tick_ms,
                             blit=False, cache_frame_data=False, repeat=False)
        print(f"Rendering {args.save_seconds:.1f}s ({n_save_frames} frames) to {args.save} using {writer}...")
        anim.save(args.save, writer=writer, fps=1000 / tick_ms)
        print(f"Saved {args.save}")
    else:
        anim = FuncAnimation(fig, tick, interval=50, blit=False, cache_frame_data=False)
        # Keep a strong reference so the animation is not garbage-collected
        # before plt.show() pumps its first frame.
        fig._iscas_demo_anim = anim
        plt.show()


if __name__ == "__main__":
    main()
