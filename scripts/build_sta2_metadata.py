"""Build verified per-video presentation-timestamp indexes."""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import index_videos


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video-info",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video-id-column",
        default="video_id",
    )
    parser.add_argument(
        "--fps-column",
        default="fps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    with args.annotations.open() as stream:
        video_ids = sorted({
            row["video_id"].upper()
            for row in csv.DictReader(stream)
        })

    with args.video_info.open() as stream:
        info_rows = list(csv.DictReader(stream))

    if (
        not info_rows
        or args.video_id_column not in info_rows[0]
        or args.fps_column not in info_rows[0]
    ):
        raise ValueError(
            "Inspect video-info columns and pass the correct "
            "--video-id-column and --fps-column"
        )

    video_info = {}

    for row in info_rows:
        video_id = row[args.video_id_column].upper()

        if video_id in video_info:
            raise ValueError(
                f"Duplicate video-info row: {video_id}"
            )

        video_info[video_id] = row

    videos = index_videos(args.raw_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "annotations_sha256": sha256_file(args.annotations),
        "video_info_sha256": sha256_file(args.video_info),
        "time_origin": "source PTS; verified zero origin",
        "videos": {},
    }

    for number, video_id in enumerate(video_ids, start=1):
        if video_id not in video_info:
            raise ValueError(
                f"Missing supplied timing metadata: {video_id}"
            )

        supplied_fps = Fraction(
            video_info[video_id][args.fps_column]
        )

        if supplied_fps <= 0:
            raise ValueError(f"Invalid FPS: {video_id}")

        item = {
            "status": "video_error",
            "supplied_fps": str(supplied_fps),
        }

        path = videos.get(video_id)

        if path is None:
            item["error"] = "missing_file"

        else:
            stat = path.stat()

            item.update(
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )

            try:
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise OSError("No video stream")

                    stream = container.streams.video[0]
                    time_base = stream.time_base

                    pts = []

                    for frame in container.decode(stream):
                        if frame.pts is None:
                            raise ValueError(
                                f"Frame without PTS: {video_id}"
                            )

                        pts.append(frame.pts)

                pts = np.asarray(pts, dtype=np.int64)

                if len(pts) < 2:
                    raise OSError(
                        "Fewer than two readable frames"
                    )

                # Do not invent an annotation/media time-origin mapping.
                if pts[0] != 0:
                    raise ValueError(
                        f"Nonzero timestamp origin: {video_id}. "
                        "Resolve the annotation/media mapping first."
                    )

                if np.any(np.diff(pts) <= 0):
                    raise ValueError(
                        f"Nonmonotonic timestamps: {video_id}"
                    )

                measured_fps = (
                    (len(pts) - 1)
                    / float(
                        (int(pts[-1]) - int(pts[0]))
                        * time_base
                    )
                )

                # Allows normal rounding in a published decimal FPS.
                # This tolerance does not relax any sampling cutoff.
                if (
                    abs(
                        measured_fps / float(supplied_fps)
                        - 1.0
                    )
                    > 0.002
                ):
                    raise ValueError(
                        f"FPS mismatch for {video_id}: "
                        f"supplied={supplied_fps}, "
                        f"measured={measured_fps}"
                    )

                timeline_path = (
                    args.output_dir / f"{video_id}.npz"
                )

                np.savez_compressed(
                    timeline_path,
                    pts=pts,
                )

                item.update(
                    status="ok",
                    timeline=timeline_path.name,
                    timeline_sha256=sha256_file(timeline_path),
                    time_base=str(time_base),
                    frame_count=len(pts),
                    measured_average_fps=measured_fps,
                )

            except (
                OSError,
                EOFError,
                av.error.FFmpegError,
            ) as error:
                item["error"] = str(error)

        manifest["videos"][video_id] = item

        output = args.output_dir / "manifest.json"
        temporary = output.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary.replace(output)

        print(
            f"{number}/{len(video_ids)} "
            f"{video_id}: {item['status']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
    