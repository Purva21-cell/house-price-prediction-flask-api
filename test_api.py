import numpy as np
import pandas as pd
import pytest
from app import create_app


class DummyModel:
    feature_names_in_ = np.array(["area", "bedrooms"])
    n_features_in_ = 2

    def predict(self, frame):
        assert list(frame.columns) == ["area", "bedrooms"]
        return np.array([float(frame.iloc[0]["area"]) * 100 + float(frame.iloc[0]["bedrooms"])])


@pytest.fixture()
def client():
    app = create_app(model=DummyModel())
    app.config.update(TESTING=True)
    return app.test_client()


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_prediction_with_named_features(client):
    response = client.post(
        "/predict",
        json={"features": {"bedrooms": 3, "area": 1000}}
    )
    assert response.status_code == 200
    assert response.get_json()["predicted_price"] == 100003.0


def test_missing_features(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_feature_mismatch(client):
    response = client.post(
        "/predict",
        json={"features": {"area": 1000, "wrong": 2}}
    )
    assert response.status_code == 400


def test_wrong_content_type(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 415
