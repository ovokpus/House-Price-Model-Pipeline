"""Build a /predict body and call the house-price API.

The Streamlit page imports this module. Tests do too, so the HTTP calls
stay here and the page file only draws widgets.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

# Same five labels the API accepts. Kept here so the UI image does not
# need the training package installed.
OCEAN_PROXIMITY_VALUES: tuple[str, ...] = (
    "<1H OCEAN",
    "INLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "ISLAND",
)

# A real district from housing.csv (Near Bay, recorded median $281,500).
# The form opens on these numbers so a first click scores something real.
EXAMPLE_DISTRICT: dict[str, Any] = {
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


class ApiError(Exception):
    """The API was unreachable or refused the district."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def api_base_url() -> str:
    """Base URL of the FastAPI service. Compose sets API_URL to http://api:8000."""
    return os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


def build_payload(
    *,
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float | None,
    bedrooms_known: bool,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
) -> dict[str, Any]:
    """Match ``HousingFeatures``. Unknown bedrooms are JSON null, not zero."""
    if ocean_proximity not in OCEAN_PROXIMITY_VALUES:
        raise ValueError(
            "ocean_proximity must be one of " + ", ".join(OCEAN_PROXIMITY_VALUES)
        )
    if households <= 0:
        raise ValueError("households must be greater than zero")

    bedrooms: float | None
    if bedrooms_known:
        if total_bedrooms is None:
            raise ValueError("total_bedrooms is required unless the count is unknown")
        bedrooms = float(total_bedrooms)
    else:
        bedrooms = None

    return {
        "longitude": float(longitude),
        "latitude": float(latitude),
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": bedrooms,
        "population": float(population),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": ocean_proximity,
    }


def fetch_health(api_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """GET /health. Raises ApiError when the process is down or has no model."""
    status, body = _read_json("GET", f"{api_url.rstrip('/')}/health", None, timeout)
    if status >= 400:
        raise ApiError(_format_error(status, body), status)
    if not isinstance(body, dict):
        raise ApiError(f"Unexpected health payload from {api_url}")
    return body


def score_district(
    api_url: str,
    payload: dict[str, Any],
    timeout: float = 20.0,
) -> dict[str, Any]:
    """POST /predict and return the JSON body."""
    status, body = _read_json(
        "POST",
        f"{api_url.rstrip('/')}/predict",
        payload,
        timeout,
    )
    if status >= 400:
        raise ApiError(_format_error(status, body), status)
    if not isinstance(body, dict) or "median_house_value" not in body:
        raise ApiError("The API response did not include median_house_value")
    return body


def _read_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, _parse_body(raw)
    except urllib.error.URLError as exc:
        raise ApiError(f"Could not reach the API at {url}: {exc.reason}") from exc
    return status, _parse_body(raw)


def _parse_body(raw: str) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"detail": raw}


def _format_error(status: int, body: Any) -> str:
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(piece) for piece in item.get("loc", []))
                message = item.get("msg", item)
                parts.append(f"{loc}: {message}" if loc else str(message))
            else:
                parts.append(str(item))
        text = "; ".join(parts)
    else:
        text = str(detail)
    return f"The API returned {status}: {text}"
