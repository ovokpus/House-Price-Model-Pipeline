"""HTTP API for district median house-value predictions.

``GET /health`` reports whether a bundle is loaded.
``POST /predict`` scores one district.
``GET /metadata`` returns the training summary stored next to the model
(metrics, features, hyperparameters) so you can see what the container
actually fitted.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from house_price import __version__
from house_price.artifacts import Artifact, load_latest
from house_price.predict import predict_median_house_value
from house_price.schemas import HealthResponse, HousingFeatures, PredictionResponse

logger = logging.getLogger(__name__)


def _public_metadata(artifact: Artifact) -> dict[str, Any]:
    """Drop the bulky preprocessor lambdas from the HTTP view."""
    metadata = dict(artifact.metadata)
    preprocessor = dict(metadata.get("preprocessor") or {})
    preprocessor.pop("yeojohnson_lambdas", None)
    metadata["preprocessor"] = preprocessor
    metadata["selected_features"] = artifact.selected_features
    return metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ``latest.json`` once. A missing bundle does not crash the process."""
    try:
        app.state.artifact = load_latest()
        logger.info("Loaded house price artifact %s", app.state.artifact.version)
    except FileNotFoundError as exc:
        app.state.artifact = None
        logger.warning("%s", exc)
    yield


def create_app() -> FastAPI:
    """Build the ASGI app. Tests construct a fresh one per case."""
    app = FastAPI(
        title="House Price Prediction",
        version=__version__,
        summary="Predict California district median house values.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | JSONResponse:
        artifact: Artifact | None = getattr(app.state, "artifact", None)
        if artifact is None:
            return JSONResponse(
                status_code=503,
                content={"status": "model_not_loaded", "artifact_version": None},
            )
        return HealthResponse(
            status="ok",
            artifact_version=artifact.version,
            algorithm=str(artifact.metadata.get("algorithm")),
            selected_feature_count=len(artifact.selected_features),
        )

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        artifact: Artifact | None = getattr(app.state, "artifact", None)
        if artifact is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run `python -m house_price.train`.",
            )
        return _public_metadata(artifact)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: HousingFeatures) -> PredictionResponse:
        artifact: Artifact | None = getattr(app.state, "artifact", None)
        if artifact is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run `python -m house_price.train`.",
            )
        try:
            value = predict_median_house_value(artifact, features)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(
            median_house_value=round(value, 2),
            artifact_version=artifact.version,
        )

    return app


app = create_app()
