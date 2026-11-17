"""Vendor the reviewed upstream model closure under a dedicated namespace."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "app/vjepa_2_1/models/vision_transformer.py":
        "models/vision_transformer.py",
    "app/vjepa_2_1/models/predictor.py":
        "models/predictor.py",
    "app/vjepa_2_1/models/utils/modules.py":
        "models/utils/modules.py",
    "app/vjepa_2_1/models/utils/patch_embed.py":
        "models/utils/patch_embed.py",
    "src/masks/utils.py":
        "masks/utils.py",
    "src/utils/tensors.py":
        "utils/tensors.py",
}

REWRITES = {
    "from src.masks.utils import":
        "from src.sta2.vendor.vjepa21.masks.utils import",
    "from src.utils.tensors import":
        "from src.sta2.vendor.vjepa21.utils.tensors import",
    "from app.vjepa_2_1.models.utils.modules import":
        "from src.sta2.vendor.vjepa21.models.utils.modules import",
    "from app.vjepa_2_1.models.utils.patch_embed import":
        "from src.sta2.vendor.vjepa21.models.utils.patch_embed import",
}


def revision(path):
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
    )

    if dirty.strip():
        raise ValueError(
            f"Upstream checkout must be clean: {path}"
        )

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vjepa",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--jfaa",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    destination = (
        PROJECT_ROOT / "src/sta2/vendor/vjepa21"
    )
    provenance = (
        PROJECT_ROOT
        / "configs/sta2/source_provenance.json"
    )

    if destination.exists() or provenance.exists():
        raise ValueError(
            "Refusing to overwrite vendor files or provenance"
        )

    manifest = {
        "vjepa_url":
            "https://github.com/facebookresearch/vjepa2",
        "vjepa_revision": revision(args.vjepa),
        "jfaa_url":
            "https://github.com/CorrineQiu/JFAA",
        "jfaa_revision": revision(args.jfaa),
        "files": {},
        "import_rewrites": REWRITES,
    }

    # Read and validate everything before creating the destination.
    contents = {}

    for original, relative in FILES.items():
        source = args.vjepa / original
        text = source.read_text()

        for before, after in REWRITES.items():
            text = text.replace(before, after)

        for line in text.splitlines():
            upstream_import = line.startswith((
                "from src.",
                "import src.",
                "from app.",
                "import app.",
            ))

            if (
                upstream_import
                and "src.sta2.vendor." not in line
            ):
                raise ValueError(
                    f"Unresolved import in {original}: {line}"
                )

        contents[relative] = text

        manifest["files"][original] = {
            "source_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            "vendored_sha256": hashlib.sha256(
                text.encode()
            ).hexdigest(),
        }

    jfaa_license = args.jfaa / "LICENSE"

    if not jfaa_license.is_file():
        raise ValueError(
            "Review the JFAA license location first"
        )

    reference_config = (
        args.jfaa
        / "configs/jfaa_vjepa2_vitG_train.yaml"
    )

    manifest["jfaa_config_sha256"] = hashlib.sha256(
        reference_config.read_bytes()
    ).hexdigest()

    for relative, text in contents.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    directories = [destination]
    directories.extend(
        path
        for path in destination.rglob("*")
        if path.is_dir()
    )

    for directory in directories:
        (directory / "__init__.py").touch()

    (destination.parent / "__init__.py").touch()

    for name in ("LICENSE", "APACHE-LICENSE"):
        source = args.vjepa / name

        if source.exists():
            shutil.copyfile(
                source,
                destination / name,
            )

    shutil.copyfile(
        jfaa_license,
        destination / "JFAA-LICENSE",
    )
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(provenance)


if __name__ == "__main__":
    main()