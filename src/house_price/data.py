"""Load the California housing districts used by the research notebooks.

The file is the StatLib California housing set redistributed with
Aurélien Géron's *Hands-On Machine Learning* examples. A copy of the
tarball already lives in this repository, so training does not need the
network. If that copy is missing, we download the same archive.
"""

from __future__ import annotations

import io
import logging
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from house_price.config import (
    HOUSING_TGZ_URL,
    RAW_FEATURE_COLUMNS,
    TARGET,
    artifact_dir,
    housing_tgz_path,
)

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in (*RAW_FEATURE_COLUMNS, TARGET) if column != "ocean_proximity"
)


def _download_housing_tgz(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Local housing archive not found; downloading %s", HOUSING_TGZ_URL)
    urllib.request.urlretrieve(HOUSING_TGZ_URL, destination)
    return destination


def resolve_housing_tgz() -> Path:
    """Locate ``housing.tgz``, downloading it into the artifact dir if needed."""
    local = housing_tgz_path()
    if local is not None:
        return local
    return _download_housing_tgz(artifact_dir() / "data" / "housing.tgz")


def load_housing_data(tgz_path: Path | None = None) -> pd.DataFrame:
    """Read ``housing.csv`` out of the tarball.

    Rows stay in file order. The notebooks split that order with a fixed
    ``random_state``, so shuffling here would train a different model.
    """
    archive = Path(tgz_path) if tgz_path is not None else resolve_housing_tgz()
    if not archive.is_file():
        raise FileNotFoundError(f"Housing archive not found: {archive}")

    with tarfile.open(archive, "r:gz") as bundle:
        try:
            handle = bundle.extractfile("housing.csv")
        except KeyError as exc:
            raise ValueError(f"{archive} does not contain housing.csv") from exc
        if handle is None:
            raise ValueError(f"{archive} does not contain a readable housing.csv")
        raw = handle.read()

    frame = pd.read_csv(io.BytesIO(raw))
    expected = list(RAW_FEATURE_COLUMNS) + [TARGET]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Housing file is missing columns: {missing}")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ocean_proximity"] = frame["ocean_proximity"].astype("object")
    return frame.loc[:, expected].copy()
