"""Checks for the deterministic feature steps copied from the analysis notebook."""

from __future__ import annotations

import pandas as pd

from house_price.data import load_housing_data
from house_price.features import (
    assign_age_group,
    assign_income_bracket,
    drop_capped_targets,
    enrich_features,
)


def test_age_and_income_cuts_match_the_notebook() -> None:
    ages = pd.Series([17, 18, 30, 31, 50, 51])
    assert assign_age_group(ages).tolist() == [
        "minor",
        "young",
        "young",
        "middle-aged",
        "middle-aged",
        "elderly",
    ]

    incomes = pd.Series([5, 5.01, 10, 10.01])
    assert assign_income_bracket(incomes).tolist() == [
        "low-income",
        "middle-income",
        "middle-income",
        "high-income",
    ]


def test_ratio_features_use_household_counts() -> None:
    raw = pd.DataFrame(
        [
            {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY",
                "median_house_value": 452600.0,
            }
        ]
    )
    enriched = enrich_features(raw)
    assert enriched.loc[0, "rooms_per_household"] == round(880 / 126)
    assert enriched.loc[0, "bedrooms_per_household"] == round(129 / 126)
    assert enriched.loc[0, "population_per_household"] == round(322 / 126)
    assert enriched.loc[0, "median_income_per_household"] == 8.3252 / 126
    assert enriched.loc[0, "income_bracket"] == "middle-income"
    assert enriched.loc[0, "age_group"] == "middle-aged"


def test_cap_drops_half_million_districts() -> None:
    frame = pd.DataFrame({"median_house_value": [100000, 499999, 500000, 500001]})
    kept = drop_capped_targets(frame)
    assert kept["median_house_value"].tolist() == [100000, 499999]


def test_loader_returns_the_full_housing_file() -> None:
    frame = load_housing_data()
    assert len(frame) == 20640
    assert frame["ocean_proximity"].nunique() == 5
    assert frame["total_bedrooms"].isna().sum() == 207
