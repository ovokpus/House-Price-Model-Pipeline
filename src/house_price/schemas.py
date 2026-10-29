"""Request and response bodies for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from house_price.config import OCEAN_PROXIMITY_VALUES

OceanProximity = Literal["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]


class HousingFeatures(BaseModel):
    """One census district, in the same units as ``housing.csv``.

    ``median_income`` is the dataset's scaled income (roughly US dollars
    divided by $10,000), not raw dollars. ``ocean_proximity`` must be one
    of the five labels in the 1990 file.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "longitude": -122.26,
                    "latitude": 37.85,
                    "housing_median_age": 52.0,
                    "total_rooms": 2202.0,
                    "total_bedrooms": 434.0,
                    "population": 910.0,
                    "households": 402.0,
                    "median_income": 3.2031,
                    "ocean_proximity": "NEAR BAY",
                }
            ]
        }
    )

    longitude: float = Field(..., ge=-180, le=180, description="District longitude.")
    latitude: float = Field(..., ge=-90, le=90, description="District latitude.")
    housing_median_age: float = Field(
        ..., ge=0, description="Median age of the housing stock, in years."
    )
    total_rooms: float = Field(..., ge=0, description="Total rooms in the district.")
    total_bedrooms: float | None = Field(
        None,
        ge=0,
        description="Total bedrooms. Omit or null when the count is unknown.",
    )
    population: float = Field(..., ge=0, description="District population.")
    households: float = Field(..., gt=0, description="Number of households.")
    median_income: float = Field(
        ...,
        ge=0,
        description="Median income in the dataset's scaled units (about $10,000s).",
    )
    ocean_proximity: OceanProximity = Field(
        ...,
        description=f"One of: {', '.join(OCEAN_PROXIMITY_VALUES)}.",
    )


class PredictionResponse(BaseModel):
    """Predicted median house value for the district, in dollars."""

    median_house_value: float = Field(..., description="Predicted median value in dollars.")
    artifact_version: str = Field(..., description="Training run that produced the model.")


class HealthResponse(BaseModel):
    """Liveness of the process and whether a model bundle is loaded."""

    status: str
    artifact_version: str | None = None
    algorithm: str | None = None
    selected_feature_count: int | None = None
