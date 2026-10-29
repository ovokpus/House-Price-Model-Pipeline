"""The form's payload and API client, without starting Streamlit."""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from housing_form import ApiError, build_payload, fetch_health, score_district


class _Body:
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_build_payload_sends_null_bedrooms_when_unknown() -> None:
    payload = build_payload(
        longitude=-122.26,
        latitude=37.85,
        housing_median_age=52,
        total_rooms=2202,
        total_bedrooms=434,
        bedrooms_known=False,
        population=910,
        households=402,
        median_income=3.2031,
        ocean_proximity="NEAR BAY",
    )
    assert payload["total_bedrooms"] is None
    assert payload["ocean_proximity"] == "NEAR BAY"
    assert payload["households"] == 402


def test_build_payload_rejects_a_bad_ocean_label() -> None:
    with pytest.raises(ValueError):
        build_payload(
            longitude=-122.0,
            latitude=37.0,
            housing_median_age=10,
            total_rooms=100,
            total_bedrooms=20,
            bedrooms_known=True,
            population=50,
            households=10,
            median_income=3,
            ocean_proximity="MOUNTAIN",
        )


def test_score_district_reads_the_dollar_value(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float = 0) -> _Body:
        assert request.get_method() == "POST"
        assert request.full_url == "http://api:8000/predict"
        sent = json.loads(request.data.decode("utf-8"))
        assert sent["ocean_proximity"] == "INLAND"
        return _Body(200, {"median_house_value": 180000.5, "artifact_version": "vtest"})

    monkeypatch.setattr("housing_form.urllib.request.urlopen", fake_urlopen)
    body = score_district(
        "http://api:8000",
        {"ocean_proximity": "INLAND"},
    )
    assert body["median_house_value"] == 180000.5
    assert body["artifact_version"] == "vtest"


def test_fetch_health_surfaces_a_down_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float = 0) -> _Body:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("housing_form.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ApiError, match="Could not reach"):
        fetch_health("http://127.0.0.1:8000")
