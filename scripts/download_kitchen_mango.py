"""Download the downscaled EPIC-KITCHENS videos from KU Leuven ManGO."""

import os
import ssl
from pathlib import Path

from irods.session import iRODSSession
from mango_auth import iinit
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BASE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "EPIC-KITCHENS" / "videos_640x360"
)
IRODS_BASE_PATH = "/set/home/ciis-lab/datasets/epic-kitchens/videos_640x360"


def main() -> None:
    """Authenticate with ManGO and download videos that are not present yet."""
    r_number = input("Enter your R-number: ")
    iinit(r_number, "set", "set.irods.icts.kuleuven.be")

    env_file = os.environ.get(
        "IRODS_ENVIRONMENT_FILE",
        str(Path.home() / ".irods" / "irods_environment.json"),
    )
    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    LOCAL_BASE_PATH.mkdir(parents=True, exist_ok=True)

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
            participant_path = LOCAL_BASE_PATH / participant
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
                    if not local_video.exists():
                        session.data_objects.get(remote_video, str(local_video))


if __name__ == "__main__":
    main()

