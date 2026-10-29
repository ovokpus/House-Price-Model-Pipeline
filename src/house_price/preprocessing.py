"""Learned transforms from the feature-engineering notebook.

Fit this on the training split only. ``transform`` replays the stored
means, Yeo-Johnson lambdas, category maps, and scaler. It does not look
at the request's target, because a request does not have one.

The step order matches ``feature-engineering.ipynb``:

1. Fill missing categories (frequent mode, or the label ``"Missing"``).
2. Add a missing indicator for numeric columns that had gaps in training,
   then fill those gaps with the training mean.
3. Yeo-Johnson on the continuous columns, except housing age and the two
   columns the notebook chose to binarize. Lambdas come from the train split.
4. Turn the skewed columns into 0/1 at a threshold of 1.
5. Map income brackets to 1/2/3.
6. Collapse categories rarer than 1% of training rows into ``"Rare"``.
7. Encode the remaining string columns by ascending mean log-price.
8. Standard-scale every column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
from sklearn.preprocessing import StandardScaler

from house_price.config import (
    DROPPED_COLUMNS,
    INCOME_BRACKET_MAPPING,
    MODEL_COLUMNS,
    RARE_CATEGORY_FRACTION,
    SKEWED_COLUMNS,
    TARGET,
)
from house_price.features import enrich_features


def model_matrix(enriched: pd.DataFrame) -> pd.DataFrame:
    """Select the columns the feature-engineering notebook actually models."""
    missing = [column for column in MODEL_COLUMNS if column not in enriched.columns]
    if missing:
        raise ValueError(f"Enriched frame is missing model columns: {missing}")
    return enriched.loc[:, list(MODEL_COLUMNS)].copy()


class HousingPreprocessor:
    """Train-set state for imputation, encoding, and scaling."""

    def fit(self, features: pd.DataFrame, y_log: pd.Series) -> HousingPreprocessor:
        """Learn every parameter from the training feature matrix.

        ``features`` is the unscaled model matrix (output of ``model_matrix``).
        ``y_log`` is ``log(median_house_value)``, already aligned row-for-row
        with ``features``. Category integers are ordered by that log target,
        which is what the notebook does.
        """
        frame = features.reset_index(drop=True).copy()
        target = pd.Series(y_log.to_numpy(), name=TARGET)
        if len(frame) != len(target):
            raise ValueError("Features and log-target must have the same number of rows")

        self.categorical_columns_: list[str] = [
            column for column in frame.columns if frame[column].dtype == object
        ]
        self.numeric_columns_: list[str] = [
            column for column in frame.columns if column not in self.categorical_columns_
        ]

        self._fit_categorical_imputer(frame)
        frame = self._apply_categorical_imputer(frame)

        self._fit_numeric_imputer(frame)
        frame = self._apply_numeric_imputer(frame)

        self.discrete_columns_: list[str] = [
            column
            for column in self.numeric_columns_
            if len(frame[column].unique()) < 20
        ]
        continuous = [
            column
            for column in self.numeric_columns_
            if column not in self.discrete_columns_ + ["longitude", "latitude"]
        ]
        self.skewed_columns_: list[str] = [
            column for column in SKEWED_COLUMNS if column in continuous
        ]
        self.yeojohnson_columns_: list[str] = [
            column
            for column in continuous
            if column not in self.skewed_columns_ and column != "housing_median_age"
        ]

        self.yeojohnson_lambdas_: dict[str, float] = {}
        for column in self.yeojohnson_columns_:
            transformed, lam = yeojohnson(frame[column].astype(float).to_numpy())
            frame[column] = transformed
            self.yeojohnson_lambdas_[column] = float(lam)

        frame = self._binarize_skewed(frame)
        frame = self._map_income_bracket(frame)

        self.other_categorical_columns_: list[str] = [
            column for column in self.categorical_columns_ if column != "income_bracket"
        ]
        self.frequent_labels_: dict[str, list[str]] = {}
        self.ordinal_maps_: dict[str, dict[str, int]] = {}
        for column in self.other_categorical_columns_:
            frequent = _frequent_labels(frame, column, RARE_CATEGORY_FRACTION)
            self.frequent_labels_[column] = frequent
            frame[column] = np.where(frame[column].isin(frequent), frame[column], "Rare")
            ordering = (
                pd.DataFrame({column: frame[column], TARGET: target.to_numpy()})
                .groupby(column)[TARGET]
                .mean()
                .sort_values()
                .index
            )
            mapping = {str(label): index for index, label in enumerate(ordering)}
            self.ordinal_maps_[column] = mapping
            frame[column] = frame[column].map(mapping)

        self.columns_: list[str] = list(frame.columns)
        numeric = frame.astype(float)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(numeric)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transforms and return a scaled frame."""
        self._check_fitted()
        frame = features.reset_index(drop=True).copy()
        missing = [column for column in MODEL_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Input is missing model columns: {missing}")
        frame = frame.loc[:, list(MODEL_COLUMNS)].copy()

        frame = self._apply_categorical_imputer(frame)
        frame = self._apply_numeric_imputer(frame)
        for column, lam in self.yeojohnson_lambdas_.items():
            frame[column] = yeojohnson(frame[column].astype(float).to_numpy(), lmbda=lam)
        frame = self._binarize_skewed(frame)
        frame = self._map_income_bracket(frame)
        for column in self.other_categorical_columns_:
            frequent = set(self.frequent_labels_[column])
            frame[column] = np.where(frame[column].isin(frequent), frame[column], "Rare")
            frame[column] = frame[column].map(self.ordinal_maps_[column])
            if frame[column].isnull().any():
                # ``Rare`` is in the map whenever training saw a rare label.
                # An all-frequent column has no ``Rare`` key; park unknowns at 0.
                fallback = self.ordinal_maps_[column].get("Rare", 0)
                frame[column] = frame[column].fillna(fallback)

        numeric = frame.loc[:, self.columns_].astype(float)
        scaled = self.scaler_.transform(numeric)
        return pd.DataFrame(scaled, columns=self.columns_)

    def transform_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Enrich a request-shaped frame, then scale it."""
        enriched = enrich_features(raw)
        return self.transform(model_matrix(enriched))

    def _fit_categorical_imputer(self, frame: pd.DataFrame) -> None:
        self.string_missing_columns_: list[str] = []
        self.category_modes_: dict[str, str] = {}
        for column in self.categorical_columns_:
            missing_rate = float(frame[column].isnull().mean())
            mode = frame[column].mode(dropna=True)
            if not mode.empty:
                self.category_modes_[column] = str(mode.iloc[0])
            if missing_rate > 0.1:
                self.string_missing_columns_.append(column)
            elif missing_rate > 0 and column in self.category_modes_:
                # Mode fill is only learned for columns that were actually sparse.
                # The stored mode still covers a request that arrives empty.
                continue

        self.frequent_fill_columns_: list[str] = [
            column
            for column in self.categorical_columns_
            if column not in self.string_missing_columns_
            and float(frame[column].isnull().mean()) > 0
        ]

    def _apply_categorical_imputer(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.categorical_columns_:
            if column not in out.columns:
                continue
            if column in self.string_missing_columns_:
                out[column] = out[column].fillna("Missing")
            elif column in self.frequent_fill_columns_ or out[column].isnull().any():
                fill = self.category_modes_.get(column, "Missing")
                out[column] = out[column].fillna(fill)
        return out

    def _fit_numeric_imputer(self, frame: pd.DataFrame) -> None:
        self.numeric_means_: dict[str, float] = {}
        self.numeric_indicator_columns_: list[str] = []
        for column in self.numeric_columns_:
            self.numeric_means_[column] = float(frame[column].mean())
            if int(frame[column].isnull().sum()) > 0:
                self.numeric_indicator_columns_.append(column)

    def _apply_numeric_imputer(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.numeric_indicator_columns_:
            out[f"{column}_na"] = np.where(out[column].isnull(), 1, 0)
        for column, mean in self.numeric_means_.items():
            if column in out.columns:
                out[column] = out[column].fillna(mean)
        return out

    def _binarize_skewed(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.skewed_columns_:
            out[column] = np.where(out[column].astype(float) <= 1, 0, 1)
        return out

    def _map_income_bracket(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "income_bracket" not in out.columns:
            return out
        out["income_bracket"] = out["income_bracket"].map(INCOME_BRACKET_MAPPING)
        if out["income_bracket"].isnull().any():
            raise ValueError(
                "income_bracket could not be mapped onto low, middle, and high"
            )
        return out

    def _check_fitted(self) -> None:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("HousingPreprocessor must be fit before transform")


def _frequent_labels(frame: pd.DataFrame, column: str, rare_fraction: float) -> list[str]:
    """Labels that cover more than ``rare_fraction`` of training rows."""
    shares = frame.groupby(column)[column].count() / len(frame)
    kept = shares[shares > rare_fraction].index
    return [str(label) for label in kept]


def describe_preprocessor(preprocessor: HousingPreprocessor) -> dict[str, Any]:
    """JSON-friendly view of the learned parameters, excluding the scaler weights."""
    return {
        "categorical_columns": preprocessor.categorical_columns_,
        "numeric_columns": preprocessor.numeric_columns_,
        "numeric_indicator_columns": preprocessor.numeric_indicator_columns_,
        "yeojohnson_columns": preprocessor.yeojohnson_columns_,
        "yeojohnson_lambdas": preprocessor.yeojohnson_lambdas_,
        "skewed_columns": preprocessor.skewed_columns_,
        "frequent_labels": preprocessor.frequent_labels_,
        "ordinal_maps": preprocessor.ordinal_maps_,
        "scaled_columns": preprocessor.columns_,
        "dropped_before_matrix": list(DROPPED_COLUMNS),
    }
