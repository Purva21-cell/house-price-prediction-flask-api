# House Price Prediction API

## Purpose
A Flask REST API that loads an existing trained house-price regression estimator
(or a complete preprocessing + estimator pipeline), accepts JSON features, and
returns a predicted price.

## Project layout
```text
House_Price_Flask_Revision/
├── app.py
├── model.joblib             # add your own trained model; not included
├── requirements.txt
├── README.md
├── test_api.py
└── test_api.ipynb
```

## 1. Set up the environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## 2. Add the trained model
Copy your own trained `.joblib` or `.pkl` file into this folder and name it
`model.joblib`. If preprocessing (encoding/scaling) was used during training,
save and deploy the complete fitted Pipeline, not just the final estimator.

Alternatively set `HOUSE_PRICE_MODEL` to the absolute path of your model file.

## 3. Run the API
```bash
python app.py
```
Default base URL: `http://127.0.0.1:5000`

## 4. Endpoints
- `GET /` – API information
- `GET /health` – health check
- `POST /predict` – prediction endpoint

### JSON input
If the fitted model has `feature_names_in_`, send a JSON object using the exact
feature names:
```json
{
  "features": {
    "actual_training_column_1": 1200,
    "actual_training_column_2": 3
  }
}
```

If feature names are not stored, send a numeric list in exact training order:
```json
{"features": [1200, 3]}
```
The example names and values are placeholders. Replace them with actual columns
and valid sample values from your training dataset.

### Successful response
```json
{"status": "success", "predicted_price": 250000.0}
```
The number above is illustrative, not a real model output.

## 5. Test with curl
Health:
```bash
curl http://127.0.0.1:5000/health
```
Prediction (replace keys and values with your actual model inputs):
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"features\":{\"actual_training_column_1\":1200,\"actual_training_column_2\":3}}"
```

## 6. Automated tests
Run:
```bash
pytest -q
```
The included tests use a small dummy estimator to test routes and validation.
They do not replace testing with your actual trained model.

## Notes
- The app does not invent a model or training feature names.
- Joblib files should only be loaded from trusted sources.
- Keep `debug=False` outside local development.
