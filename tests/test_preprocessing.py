"""The learned preprocessor has to be stable between fit and a later request."""

from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.data import load_housing_data
from house_price.features import enrich_features
from house_price.preprocessing import HousingPreprocessor, model_matrix
from house_price.train import make_splits


def test_transform_matches_training_rows_and_has_no_nulls() -> None:
    raw = load_housing_data()
    splits = make_splits(raw)
    y_log = pd.Series(np.log(splits["y_train"].to_numpy(dtype=float)))
    preprocessor = HousingPreprocessor().fit(splits["x_train"], y_log)

    train_scaled = preprocessor.transform(splits["x_train"])
    holdout_scaled = preprocessor.transform(splits["x_holdout"])

    assert list(train_scaled.columns) == list(holdout_scaled.columns)
    assert not train_scaled.isnull().any().any()
    assert not holdout_scaled.isnull().any().any()
    # Standard scaling is fit on the train split, so those columns sit near 0.
    assert train_scaled.mean().abs().max() < 1e-6


def test_island_is_treated_as_rare_like_the_notebook() -> None:
    raw = load_housing_data()
    enriched = enrich_features(raw)
    matrix = model_matrix(enriched)
    y_log = pd.Series(np.log(enriched["median_house_value"].clip(lower=1).to_numpy(dtype=float)))
    preprocessor = HousingPreprocessor().fit(matrix, y_log)

    island = matrix.iloc[[0]].copy()
    island["ocean_proximity"] = "ISLAND"
    inland = matrix.iloc[[0]].copy()
    inland["ocean_proximity"] = "INLAND"
    scored_island = preprocessor.transform(island)
    scored_inland = preprocessor.transform(inland)
    assert scored_island["ocean_proximity"].iloc[0] != scored_inland["ocean_proximity"].iloc[0]
    assert "Rare" in preprocessor.ordinal_maps_["ocean_proximity"]
