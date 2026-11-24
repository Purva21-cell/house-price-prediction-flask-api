"""House Price Prediction REST API.

Place your trained model or complete preprocessing pipeline at MODEL_PATH,
or set the HOUSE_PRICE_MODEL environment variable to its path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
MODEL_PATH = Path(os.getenv("HOUSE_PRICE_MODEL", str(DEFAULT_MODEL_PATH))).expanduser()


def load_model(model_path: Path) -> Any:
    """Load the previously trained estimator or preprocessing pipeline."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Copy your trained .joblib/.pkl model here or set HOUSE_PRICE_MODEL."
        )
    return joblib.load(model_path)


def create_app(model: Any | None = None) -> Flask:
    """Create the Flask application; model injection also enables unit testing."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    loaded_model = model if model is not None else load_model(MODEL_PATH)
    feature_names = getattr(loaded_model, "feature_names_in_", None)
    expected_count = getattr(loaded_model, "n_features_in_", None)

    app.extensions["prediction_model"] = loaded_model
    app.extensions["feature_names"] = (
        [str(name) for name in feature_names] if feature_names is not None else None
    )
    app.extensions["expected_feature_count"] = (
        int(expected_count) if expected_count is not None else None
    )

    @app.get("/")
    def home():
        return jsonify({
            "status": "success",
            "message": "House Price Prediction API is running.",
            "endpoints": {
                "health": "GET /health",
                "prediction": "POST /predict"
            }
        }), 200

    @app.get("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "model_loaded": True
        }), 200

    def prepare_input(features: Any):
        names = app.extensions["feature_names"]
        count = app.extensions["expected_feature_count"]

        if isinstance(features, dict):
            if names is None:
                raise ValueError(
                    "This model has no stored feature names. "
                    "Send 'features' as an ordered numeric list."
                )
            missing = [name for name in names if name not in features]
            extra = [name for name in features if name not in names]
            if missing or extra:
                raise ValueError(
                    f"Feature mismatch. Missing: {missing}; unexpected: {extra}"
                )
            # Preserve training-time column order.
            return pd.DataFrame([[features[name] for name in names]], columns=names)

        if isinstance(features, list):
            if count is not None and len(features) != count:
                raise ValueError(
                    f"Expected {count} features, received {len(features)}."
                )
            if not features:
                raise ValueError("'features' list cannot be empty.")
            try:
                values = np.asarray(features, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("List features must contain numeric values.") from exc
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("Features must be a 1-D list of finite numbers.")
            if names is not None:
                return pd.DataFrame([values], columns=names)
            return values.reshape(1, -1)

        raise ValueError("'features' must be a JSON object or numeric list.")

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json."}), 415

        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify({"error": "Request body contains invalid JSON."}), 400

        if not isinstance(payload, dict) or "features" not in payload:
            return jsonify({"error": "JSON body must contain a 'features' field."}), 400

        try:
            model_input = prepare_input(payload["features"])
            prediction = loaded_model.predict(model_input)
            prediction_array = np.asarray(prediction).reshape(-1)

            if prediction_array.size != 1:
                raise ValueError("The model must return exactly one prediction.")

            predicted_price = float(prediction_array[0])
            if not np.isfinite(predicted_price):
                raise ValueError("The model returned a non-finite prediction.")

            return jsonify({
                "status": "success",
                "predicted_price": predicted_price
            }), 200

        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            LOGGER.exception("Prediction failed.")
            return jsonify({"error": "Prediction failed due to an internal error."}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=False
    )
