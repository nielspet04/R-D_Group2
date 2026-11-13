"""Download the downscaled EPIC-KITCHENS videos from KU Leuven ManGO."""

import argparse
import os
import ssl
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BASE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "EPIC-KITCHENS" / "videos_640x360"
)
IRODS_BASE_PATH = "/set/home/ciis-lab/datasets/epic-kitchens/videos_640x360"


def download_video(session, remote_path: str, local_path: Path,
                   expected_size: int, overwrite: bool = False) -> bool:
    """Publish only complete downloads; preserve the previous file on failure."""
    if local_path.exists() and not overwrite:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mango-", dir=local_path.parent) as staging:
        temporary_path = Path(staging) / local_path.name
        session.data_objects.get(remote_path, str(temporary_path))
        actual_size = temporary_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Download size mismatch for {remote_path}: "
                f"expected {expected_size}, got {actual_size}"
            )
        temporary_path.replace(local_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_BASE_PATH,
                        help="Destination containing the participant folders.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Download current ManGO copies even when local files exist.")
    args = parser.parse_args()

    from irods.session import iRODSSession
    from mango_auth import iinit
    from tqdm import tqdm

    r_number = input("Enter your R-number: ")
    iinit(r_number, "set", "set.irods.icts.kuleuven.be")

    env_file = os.environ.get(
        "IRODS_ENVIRONMENT_FILE",
        str(Path.home() / ".irods" / "irods_environment.json"),
    )
    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with iRODSSession(
        irods_env_file=env_file,
        ssl_context=ssl_context,
    ) as session:
        collection = session.collections.get(IRODS_BASE_PATH)
        participants = [item.name for item in collection.subcollections]

        for participant in tqdm(
            participants,
            desc="Downloading videos",
            unit="participant",
        ):
            participant_path = args.output_dir / participant
            participant_path.mkdir(parents=True, exist_ok=True)

            remote_path = f"{IRODS_BASE_PATH}/{participant}"
            participant_collection = session.collections.get(remote_path)

            for data_object in tqdm(
                participant_collection.data_objects,
                desc=f"Downloading videos for {participant}",
                leave=False,
                unit="video",
            ):
                if data_object.name.lower().endswith(".mp4"):
                    local_video = participant_path / data_object.name
                    remote_video = f"{remote_path}/{data_object.name}"
                    download_video(
                        session, remote_video, local_video,
                        expected_size=int(data_object.size),
                        overwrite=args.overwrite,
                    )


if __name__ == "__main__":
    main()

