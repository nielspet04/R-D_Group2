"""Read-only validation of existing STA2 timelines; no video decoding or writes."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path

from build_sta2_metadata import reusable_timeline, sha256_file
from src.data_processing import index_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/EPIC_100_train.csv"))
    parser.add_argument("--video-info", type=Path, default=Path("data/raw/EPIC_100_video_info.csv"))
    args = parser.parse_args()
    manifest = json.loads((args.metadata_dir / "manifest.json").read_text())
    for key, expected in {
        "schema_version": 1,
        "complete": True,
        "time_origin": "source PTS; verified zero origin",
        "annotations_sha256": sha256_file(args.annotations),
        "video_info_sha256": sha256_file(args.video_info),
    }.items():
        if manifest.get(key) != expected:
            parser.error(f"manifest identity mismatch: {key}")
    with args.annotations.open() as stream:
        required = {row["video_id"].upper() for row in csv.DictReader(stream)}
    with args.video_info.open() as stream:
        rows = list(csv.DictReader(stream))
    supplied = {row["video_id"].upper(): Fraction(row["fps"]) for row in rows}
    if len(supplied) != len(rows):
        parser.error("duplicate video-info IDs")
    videos = manifest.get("videos", {})
    if set(videos) != required:
        parser.error("manifest video coverage differs from annotations")
    paths = index_videos(args.raw_dir)
    failures = []
    for video_id in sorted(required):
        fps = supplied.get(video_id)
        if fps is None or fps <= 0 or not reusable_timeline(
            videos[video_id], paths.get(video_id), args.metadata_dir, fps
        ):
            failures.append(video_id)
    print("Metadata:", args.metadata_dir)
    print("Statuses:", dict(Counter(row.get("status") for row in videos.values())))
    print("Verified source identities and timeline integrity:", len(required) - len(failures), "/", len(required))
    if failures:
        parser.error(f"non-reusable timelines (first 20): {', '.join(failures[:20])}; inspect or rebuild on CPU")
    print("PASS: read-only metadata checks; no videos decoded. GPU decoding remains to be tested.")


if __name__ == "__main__":
    main()
