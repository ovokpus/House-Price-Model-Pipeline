"""Constants shared by training and serving.

Values here are the ones the research notebooks actually settled on, not
a fresh hyperparameter search. See ``model-training.ipynb`` for the grid
search that picked the gradient boosting settings.
"""

from __future__ import annotations

import os
from pathlib import Path

# Original California housing columns, in the order they appear in housing.csv.
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
)

TARGET = "median_house_value"

# The census file caps recorded district medians at $500,001. The analysis
# notebook drops every row at or above $500,000 so the model does not learn
# that prices stop there.
TARGET_CAP = 500_000

# Columns the analysis notebook appends before the first train/test split.
ENGINEERED_COLUMNS: tuple[str, ...] = (
    "age_group",
    "income_bracket",
    "rooms_per_household",
    "bedrooms_per_household",
    "population_per_household",
    "median_income_per_household",
)

# Dropped after enrichment. ``age_group`` is collinear with housing age,
# and ``total_bedrooms`` / ``households`` are collinear with the per-household
# ratios. The target is removed from the feature matrix.
DROPPED_COLUMNS: tuple[str, ...] = (
    "total_bedrooms",
    "households",
    "age_group",
    TARGET,
)

# Feature matrix column order after those drops. This matches ``train.csv``
# column order in the feature-engineering notebook.
MODEL_COLUMNS: tuple[str, ...] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "population",
    "median_income",
    "ocean_proximity",
    "income_bracket",
    "rooms_per_household",
    "bedrooms_per_household",
    "population_per_household",
    "median_income_per_household",
)

# Categories observed in the 1990 California housing file.
OCEAN_PROXIMITY_VALUES: tuple[str, ...] = (
    "<1H OCEAN",
    "INLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "ISLAND",
)

# Analysis notebook: 70/30 split. Feature engineering then keeps only the
# 70% side and holds out 10% of that for the score reported in training.
OUTER_TEST_SIZE = 0.3
OUTER_RANDOM_STATE = 42
INNER_TEST_SIZE = 0.1
INNER_RANDOM_STATE = 0

# feature-selection.ipynb
LASSO_ALPHA = 0.001
LASSO_RANDOM_STATE = 0
RARE_CATEGORY_FRACTION = 0.01

# Winning model written down in model-training.ipynb after GridSearchCV.
# The service trains this configuration directly. Re-running the full grid
# (several hundred forest fits) is research work, not container startup.
GBR_N_ESTIMATORS = 300
GBR_LEARNING_RATE = 0.3
GBR_MIN_SAMPLES_LEAF = 3
GBR_RANDOM_STATE = 0

# Continuous columns the feature-engineering notebook binarizes instead of
# transforming. ``median_income_per_household`` is income (already on a
# ~0–15 scale in this file) divided by household count, so it is almost
# always below 1 and the binary column is nearly constant. Kept as-is so
# the packaged model follows the notebooks; see PRODUCTIONIZING.md.
SKEWED_COLUMNS: tuple[str, ...] = (
    "population_per_household",
    "median_income_per_household",
)

INCOME_BRACKET_MAPPING: dict[str, int] = {
    "low-income": 1,
    "middle-income": 2,
    "high-income": 3,
}

# Public copy of the same tarball already stored in this repo.
# Used only when the local file cannot be found.
HOUSING_TGZ_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.tgz"
)

RELATIVE_TGZ = Path("01_research-and-development") / "datasets" / "housing" / "housing.tgz"


def artifact_dir() -> Path:
    """Directory that holds versioned model bundles and ``latest.json``."""
    return Path(os.environ.get("ARTIFACT_DIR", "artifacts"))


def housing_tgz_path() -> Path | None:
    """Return the local housing archive if it is already on disk."""
    override = os.environ.get("HOUSING_TGZ")
    if override:
        path = Path(override)
        return path if path.is_file() else None

    candidates = [Path.cwd() / RELATIVE_TGZ]
    here = Path(__file__).resolve()
    candidates.extend(parent / RELATIVE_TGZ for parent in here.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
