"""Extract frames from a (virtual) video camera file for labelling and training.

Usage::

    python ml/training/extract_frames.py PATH/TO/video.mp4 --out ml/datasets/frames/<name>
    python ml/training/extract_frames.py video.mp4 --every-n-frames 30 --max-frames 200

Frames are written as ``frame_000001.png`` plus a ``manifest.csv`` with
``filename,frame_index,timestamp_s`` so labelling tools and the Phase 4/6
training scripts can pair pixels with capture times.

Nothing here labels water/land — this only prepares material. Coastal footage
is **operator-collected data** and is never committed to git (see .gitignore
for ``ml/datasets/``). Synthetic frames generated for tests must be labelled
synthetic everywhere they are used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Video file to sample (e.g. mp4 under DATA_DIR/videos).")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: ml/datasets/frames/<video stem>).")
    parser.add_argument("--every-n-frames", type=int, default=30, help="Keep every Nth frame (default 30).")
    parser.add_argument("--max-frames", type=int, default=500, help="Stop after this many frames (default 500).")
    parser.add_argument("--start", type=int, default=0, help="Skip frames before this index.")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    if args.every_n_frames < 1:
        parser.error("--every-n-frames must be >= 1")

    out_dir = args.out or Path("ml") / "datasets" / "frames" / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

    kept = 0
    index = 0
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "frame_index", "timestamp_s"])
        while kept < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= args.start and index % args.every_n_frames == 0:
                filename = f"frame_{kept:06d}.png"
                if not cv2.imwrite(str(out_dir / filename), frame):
                    print(f"warning: could not write {filename}", file=sys.stderr)
                    cap.release()
                    return 1
                writer.writerow([filename, index, f"{index / fps:.3f}"])
                kept += 1
            index += 1
    cap.release()
    print(f"extracted {kept} frame(s) to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
