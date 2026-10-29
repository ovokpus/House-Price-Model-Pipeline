"""Versioned model bundles.

Each training run writes a new directory and points ``latest.json`` at it.
Older directories stay put so a bad run can be rolled back by editing that
pointer (or by setting ``ARTIFACT_DIR`` at a specific version folder's parent
and a hand-written ``latest.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from house_price.config import artifact_dir as default_artifact_dir
from house_price.preprocessing import HousingPreprocessor


@dataclass
class Artifact:
    """A fitted preprocessor, the boosting model, and the run metadata."""

    version: str
    directory: Path
    model: Any
    preprocessor: HousingPreprocessor
    selected_features: list[str]
    metadata: dict[str, Any]


def new_version_id(now: datetime | None = None) -> str:
    """UTC timestamp id, for example ``v20261029T173000Z``."""
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("v%Y%m%dT%H%M%SZ")


def save_artifact(
    artifact_dir: Path,
    model: Any,
    preprocessor: HousingPreprocessor,
    selected_features: list[str],
    metadata: dict[str, Any],
    version: str | None = None,
) -> Artifact:
    """Write one version directory and update ``latest.json``."""
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    version_id = version or new_version_id()
    directory = root / version_id
    if directory.exists():
        version_id = f"{version_id}-{datetime.now(timezone.utc).strftime('%f')}"
        directory = root / version_id
    directory.mkdir(parents=True, exist_ok=False)

    metadata = dict(metadata)
    metadata["version"] = version_id
    metadata["selected_features"] = list(selected_features)

    joblib.dump(model, directory / "model.joblib")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pointer = {"version": version_id, "directory": version_id}
    (root / "latest.json").write_text(json.dumps(pointer, indent=2) + "\n")

    return Artifact(
        version=version_id,
        directory=directory,
        model=model,
        preprocessor=preprocessor,
        selected_features=list(selected_features),
        metadata=metadata,
    )


def load_artifact(directory: Path) -> Artifact:
    """Load one version directory."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No metadata.json in {directory}")
    metadata = json.loads(metadata_path.read_text())
    selected = list(metadata.get("selected_features") or [])
    if not selected:
        raise ValueError(f"{metadata_path} does not list selected_features")
    return Artifact(
        version=str(metadata.get("version") or directory.name),
        directory=directory,
        model=joblib.load(directory / "model.joblib"),
        preprocessor=joblib.load(directory / "preprocessor.joblib"),
        selected_features=selected,
        metadata=metadata,
    )


def load_latest(artifact_dir: Path | None = None) -> Artifact:
    """Load whichever version ``latest.json`` currently names."""
    root = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir()
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No model pointer at {pointer_path}. Train one with "
            "`python -m house_price.train`."
        )
    pointer = json.loads(pointer_path.read_text())
    directory = root / pointer["directory"]
    return load_artifact(directory)
