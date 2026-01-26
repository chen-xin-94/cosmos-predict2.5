#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path

ID_RE = re.compile(r"^(\d+)_chunk\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create width-concatenated GT videos from dataset annotations."
    )
    parser.add_argument(
        "--output-dir",
        default="/raid/chen.xin/repo/cosmos-predict2.5/outputs/ac_mv_448_1344_inference",
        help="Folder containing *_chunk.mp4 files and where gt/ will be written.",
    )
    parser.add_argument(
        "--ann-dir",
        default=(
            "/raid/chen.xin/repo/cosmos-predict2.5/datasets/df/"
            "avla_nov_8_merged_per_embodiment_2025-11-12/"
            "fr3_single_arm_franka_hand/annotation/test"
        ),
        help="Annotation folder containing {id}.json files.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg binary to use (default: ffmpeg).",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe binary to use (default: ffprobe).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs in gt/.",
    )
    return parser.parse_args()


def read_video_paths(json_path: Path) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos")
    if not isinstance(videos, list) or len(videos) != 3:
        raise ValueError(f"{json_path} missing 3 videos entries")
    paths = []
    for item in videos:
        path = item.get("video_path")
        if not path:
            raise ValueError(f"{json_path} has empty video_path")
        paths.append(path)
    return paths


def get_duration_seconds(ffprobe: str, video_path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def run_ffmpeg(
    ffmpeg: str,
    inputs: list[str],
    output_path: Path,
    overwrite: bool,
    target_duration: float,
    source_duration: float,
) -> None:
    args = [ffmpeg]
    if overwrite:
        args.append("-y")
    else:
        args.append("-n")
    for p in inputs:
        args.extend(["-i", p])
    if source_duration <= 0:
        raise ValueError("source duration must be > 0")
    scale = target_duration / source_duration
    args.extend(
        [
            "-filter_complex",
            f"[0:v][1:v][2:v]hstack=inputs=3,setpts=PTS*{scale:.9f}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-an",
            "-vsync",
            "vfr",
            str(output_path),
        ]
    )
    subprocess.run(args, check=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ann_dir = Path(args.ann_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for entry in sorted(output_dir.iterdir()):
        if not entry.is_file():
            continue
        match = ID_RE.match(entry.name)
        if not match:
            continue
        sample_id = match.group(1)
        chunk_path = entry
        ann_path = ann_dir / f"{sample_id}.json"
        if not ann_path.exists():
            print(f"Skip {sample_id}: missing annotation {ann_path}")
            continue
        try:
            video_paths = read_video_paths(ann_path)
        except Exception as exc:
            print(f"Skip {sample_id}: {exc}")
            continue
        out_path = gt_dir / f"{sample_id}_gt_concat.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"Skip {sample_id}: {out_path} exists")
            continue
        try:
            target_duration = get_duration_seconds(args.ffprobe, chunk_path)
            source_duration = max(
                get_duration_seconds(args.ffprobe, Path(p)) for p in video_paths
            )
        except Exception as exc:
            print(f"Skip {sample_id}: failed to read duration: {exc}")
            continue
        print(f"Concat {sample_id} -> {out_path}")
        run_ffmpeg(
            args.ffmpeg,
            video_paths,
            out_path,
            args.overwrite,
            target_duration,
            source_duration,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
