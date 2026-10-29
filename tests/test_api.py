"""HTTP contract: /health and /predict, including the empty-model case."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from house_price.api import create_app
from house_price.train import train

PREDICT_BODY = {
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


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small-but-real bundle so the API tests do not fit 300 trees."""
    root = tmp_path_factory.mktemp("artifacts")
    train(artifact_root=root, n_estimators=15, sample_rows=4000)
    return root


@pytest.fixture()
def client(artifact_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_without_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    with TestClient(create_app()) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "model_not_loaded"


def test_health_reports_the_loaded_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["artifact_version"]
    assert body["selected_feature_count"] > 0


def test_predict_returns_a_positive_dollar_value(client: TestClient) -> None:
    response = client.post("/predict", json=PREDICT_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["artifact_version"]
    assert 20_000 < body["median_house_value"] < 1_500_000


def test_predict_rejects_an_unknown_ocean_label(client: TestClient) -> None:
    payload = dict(PREDICT_BODY)
    payload["ocean_proximity"] = "MOUNTAIN"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_accepts_a_missing_bedroom_count(client: TestClient) -> None:
    payload = dict(PREDICT_BODY)
    payload["total_bedrooms"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["median_house_value"] > 0
