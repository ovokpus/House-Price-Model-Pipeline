"""Score one district from a loaded artifact.

The model predicts ``log(median_house_value)``. Callers get dollars back
via ``exp``, which is the inverse the training notebook uses (it calls the
transform Yeo-Johnson in the prose and applies ``np.log`` / ``np.exp`` in
the code).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.artifacts import Artifact
from house_price.schemas import HousingFeatures


def predict_median_house_value(artifact: Artifact, features: HousingFeatures) -> float:
    """Return the predicted district median house value in dollars."""
    frame = pd.DataFrame([features.model_dump()])
    scaled = artifact.preprocessor.transform_raw(frame)
    missing = [name for name in artifact.selected_features if name not in scaled.columns]
    if missing:
        raise RuntimeError(
            "The loaded preprocessor did not produce selected features: "
            + ", ".join(missing)
        )
    log_price = float(artifact.model.predict(scaled[artifact.selected_features])[0])
    price = float(np.exp(log_price))
    if not np.isfinite(price):
        raise RuntimeError("Model returned a non-finite house value")
    return price
