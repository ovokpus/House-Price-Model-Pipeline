"""Train the notebook's gradient boosting model and save a versioned bundle.

The search that chose these hyperparameters lives in
``model-training.ipynb``. This script does not repeat that search. It
rebuilds the feature matrix, selects columns with the same Lasso settings,
fits ``GradientBoostingRegressor(min_samples_leaf=3, n_estimators=300,
learning_rate=0.3)``, and writes the artifact the API loads.

Run it with ``python -m house_price.train``.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from house_price.artifacts import Artifact, save_artifact
from house_price.config import (
    GBR_LEARNING_RATE,
    GBR_MIN_SAMPLES_LEAF,
    GBR_N_ESTIMATORS,
    GBR_RANDOM_STATE,
    INNER_RANDOM_STATE,
    INNER_TEST_SIZE,
    LASSO_ALPHA,
    LASSO_RANDOM_STATE,
    OUTER_RANDOM_STATE,
    OUTER_TEST_SIZE,
    TARGET,
    TARGET_CAP,
    artifact_dir,
)
from house_price.data import load_housing_data
from house_price.features import drop_capped_targets, enrich_features
from house_price.preprocessing import (
    HousingPreprocessor,
    describe_preprocessor,
    model_matrix,
)

logger = logging.getLogger(__name__)


def _metrics(y_log: pd.Series, prediction_log: np.ndarray) -> dict[str, float]:
    """MSE, RMSE, and R^2 on the dollar scale, matching the training notebook."""
    actual = np.exp(y_log.to_numpy())
    predicted = np.exp(prediction_log)
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
        "n": int(len(actual)),
    }


def make_splits(raw: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Reproduce the notebook's two-step split on enriched, uncapped rows.

    First a 70/30 split (``random_state=42``). Rows on the $500,000 cap are
    removed after that split, on both sides. The feature-engineering
    notebook then ignores the 30% side and holds out 10% of the remainder
    (``random_state=0``) as the score set. We still score the outer 30%
    and store it alongside the notebook holdout.
    """
    enriched = enrich_features(raw)
    outer_train, outer_test = train_test_split(
        enriched,
        test_size=OUTER_TEST_SIZE,
        random_state=OUTER_RANDOM_STATE,
    )
    outer_train = drop_capped_targets(outer_train).reset_index(drop=True)
    outer_test = drop_capped_targets(outer_test).reset_index(drop=True)

    x_outer = model_matrix(outer_train)
    y_outer = outer_train[TARGET].astype(float)
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_outer,
        y_outer,
        test_size=INNER_TEST_SIZE,
        random_state=INNER_RANDOM_STATE,
    )
    return {
        "x_train": x_train.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "x_holdout": x_holdout.reset_index(drop=True),
        "y_holdout": y_holdout.reset_index(drop=True),
        "x_outer_test": model_matrix(outer_test).reset_index(drop=True),
        "y_outer_test": outer_test[TARGET].astype(float).reset_index(drop=True),
    }


def select_features(x_train: pd.DataFrame, y_log: pd.Series) -> list[str]:
    """Keep columns whose Lasso coefficient survives ``alpha=0.001``.

    ``SelectFromModel`` uses its default threshold (the mean absolute
    coefficient), same as ``feature-selection.ipynb``. ``max_iter`` is
    higher than that notebook's default so the fit converges on this
    matrix. If a tiny sample shrinks every coefficient to zero, every
    column is kept and the metadata records the fallback.
    """
    selector = SelectFromModel(
        Lasso(alpha=LASSO_ALPHA, random_state=LASSO_RANDOM_STATE, max_iter=10_000)
    )
    selector.fit(x_train, y_log.to_numpy())
    chosen = [column for column, keep in zip(x_train.columns, selector.get_support()) if keep]
    return chosen


def train(
    artifact_root: Path | None = None,
    n_estimators: int = GBR_N_ESTIMATORS,
    learning_rate: float = GBR_LEARNING_RATE,
    min_samples_leaf: int = GBR_MIN_SAMPLES_LEAF,
    sample_rows: int | None = None,
    tgz_path: Path | None = None,
) -> Artifact:
    """Fit the packaged pipeline and write ``artifacts/<version>/``."""
    raw = load_housing_data(tgz_path)
    if sample_rows is not None:
        if sample_rows < 500:
            raise ValueError("sample_rows must be at least 500 to fit the pipeline")
        raw = raw.sample(n=sample_rows, random_state=0).reset_index(drop=True)

    splits = make_splits(raw)
    y_train_log = pd.Series(np.log(splits["y_train"].to_numpy(dtype=float)), name=TARGET)
    y_holdout_log = pd.Series(np.log(splits["y_holdout"].to_numpy(dtype=float)), name=TARGET)
    y_outer_log = pd.Series(np.log(splits["y_outer_test"].to_numpy(dtype=float)), name=TARGET)

    preprocessor = HousingPreprocessor().fit(splits["x_train"], y_train_log)
    x_train = preprocessor.transform(splits["x_train"])
    x_holdout = preprocessor.transform(splits["x_holdout"])
    x_outer = preprocessor.transform(splits["x_outer_test"])

    selected = select_features(x_train, y_train_log)
    selection_fallback = False
    if not selected:
        selected = list(x_train.columns)
        selection_fallback = True
        logger.warning("Lasso selected no features; keeping the full scaled matrix")

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        random_state=GBR_RANDOM_STATE,
    )
    model.fit(x_train[selected], y_train_log.to_numpy())

    holdout_metrics = _metrics(y_holdout_log, model.predict(x_holdout[selected]))
    outer_metrics = _metrics(y_outer_log, model.predict(x_outer[selected]))
    hyperparameters = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "min_samples_leaf": min_samples_leaf,
        "random_state": GBR_RANDOM_STATE,
    }
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "GradientBoostingRegressor",
        "hyperparameters": hyperparameters,
        "target": TARGET,
        "target_transform": "log",
        "target_cap_exclusive": TARGET_CAP,
        "metrics": {
            "notebook_holdout": holdout_metrics,
            "outer_test": outer_metrics,
        },
        "splits": {
            "outer_test_size": OUTER_TEST_SIZE,
            "outer_random_state": OUTER_RANDOM_STATE,
            "inner_test_size": INNER_TEST_SIZE,
            "inner_random_state": INNER_RANDOM_STATE,
        },
        "feature_selection": {
            "estimator": "Lasso",
            "alpha": LASSO_ALPHA,
            "random_state": LASSO_RANDOM_STATE,
            "fallback_all_features": selection_fallback,
        },
        "n_rows_loaded": int(len(raw)),
        "n_rows_inner_train": int(len(splits["x_train"])),
        "sample_rows": sample_rows,
        "preprocessor": describe_preprocessor(preprocessor),
    }
    bundle = save_artifact(
        artifact_root or artifact_dir(),
        model=model,
        preprocessor=preprocessor,
        selected_features=selected,
        metadata=metadata,
    )
    logger.info(
        "Saved %s holdout RMSE %.0f R2 %.3f",
        bundle.version,
        holdout_metrics["rmse"],
        holdout_metrics["r2"],
    )
    return bundle


def main(argv: list[str] | None = None) -> None:
    """CLI entry point used by the Makefile and the container entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the house price model")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Where to write the versioned bundle. Defaults to $ARTIFACT_DIR or ./artifacts.",
    )
    parser.add_argument("--n-estimators", type=int, default=GBR_N_ESTIMATORS)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Train on a random subset. For tests only; omit for the real model.",
    )
    parser.add_argument("--housing-tgz", type=Path, default=None)
    args = parser.parse_args(argv)

    bundle = train(
        artifact_root=args.artifact_dir,
        n_estimators=args.n_estimators,
        sample_rows=args.sample_rows,
        tgz_path=args.housing_tgz,
    )
    holdout = bundle.metadata["metrics"]["notebook_holdout"]
    outer = bundle.metadata["metrics"]["outer_test"]
    print(f"version: {bundle.version}")
    print(f"directory: {bundle.directory}")
    print(f"selected_features ({len(bundle.selected_features)}): {', '.join(bundle.selected_features)}")
    print(
        "notebook_holdout_rmse: {rmse:.2f} r2: {r2:.4f} n: {n}".format(**holdout)
    )
    print("outer_test_rmse: {rmse:.2f} r2: {r2:.4f} n: {n}".format(**outer))


if __name__ == "__main__":
    main()
