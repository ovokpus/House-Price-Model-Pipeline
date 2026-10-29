"""Deterministic features from the exploratory notebook.

Nothing in this module is learned from a training split. The same function
runs before the train/test split and again, unchanged, on a request to
``/predict``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.config import RAW_FEATURE_COLUMNS, TARGET, TARGET_CAP

# Housing-stock age brackets from data-analysis.ipynb. The column is the
# median age of the *houses* in the district. The notebook still labeled the
# buckets as if they were people (minor / young / elderly) and later dropped
# the column. We keep that mapping so the research path stays reproducible.
def assign_age_group(housing_median_age: pd.Series) -> pd.Series:
    """Map median housing age onto the notebook's four buckets."""
    age = housing_median_age.astype(float)
    labels = np.select(
        [age < 18, age <= 30, age <= 50],
        ["minor", "young", "middle-aged"],
        default="elderly",
    )
    return pd.Series(labels, index=housing_median_age.index, dtype="object")


def assign_income_bracket(median_income: pd.Series) -> pd.Series:
    """Map median income onto low / middle / high.

    In this dataset ``median_income`` is already scaled (roughly tens of
    thousands of dollars). The notebook cuts are ``<= 5``, ``<= 10``, else
    high. A value of exactly 5 is low-income; exactly 10 is middle-income.
    """
    income = median_income.astype(float)
    labels = np.select(
        [income <= 5, income <= 10],
        ["low-income", "middle-income"],
        default="high-income",
    )
    return pd.Series(labels, index=median_income.index, dtype="object")


def enrich_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Append the ratio and bracket columns the analysis notebook adds.

    ``households`` of 0 would make the ratios infinite. Those rows become
    missing ratios and the learned imputer (fit on training data) fills them.
    The published file has no empty households; the guard is for requests.
    """
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    frame = raw.copy()
    frame["age_group"] = assign_age_group(frame["housing_median_age"])
    frame["income_bracket"] = assign_income_bracket(frame["median_income"])

    households = frame["households"].astype(float).replace(0, np.nan)
    frame["rooms_per_household"] = (frame["total_rooms"].astype(float) / households).round()
    frame["bedrooms_per_household"] = (
        frame["total_bedrooms"].astype(float) / households
    ).round()
    frame["population_per_household"] = (
        frame["population"].astype(float) / households
    ).round()
    frame["median_income_per_household"] = frame["median_income"].astype(float) / households
    return frame


def drop_capped_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove districts whose recorded median is on the $500,000 cap."""
    if TARGET not in frame.columns:
        raise ValueError(f"Cannot apply the target cap without '{TARGET}'")
    kept = frame.loc[frame[TARGET] < TARGET_CAP].copy()
    return kept
