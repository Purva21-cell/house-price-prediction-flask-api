# House Price Prediction using Flask API

## Project Overview

This project demonstrates the deployment of a trained Machine Learning model for House Price Prediction using Flask.

The application provides a REST API that accepts house-related input features in JSON format, processes the input using a trained Linear Regression model, and returns the predicted house price as a JSON response.

## Technologies Used

- Python
- Flask
- Machine Learning
- Linear Regression
- Scikit-learn
- Pandas
- NumPy
- Joblib
- REST API
- Jupyter Notebook

## Project Structure

```text
House_Price_Flask/
│
├── app.py
├── model.joblib
├── requirements.txt
├── README.md
└── test_api.ipynb
```

## Features

- Loads a previously trained Machine Learning model.
- Deploys the model using Flask.
- Provides a REST API for house price prediction.
- Accepts JSON input.
- Generates predictions using `model.predict()`.
- Returns prediction results in JSON format.
- Includes input validation and error handling.
- Provides API health-check functionality.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction-flask-api.git
```

Navigate to the project directory:

```bash
cd house-price-prediction-flask-api
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Flask Application

```bash
python app.py
```

The application will run at:

```text
http://127.0.0.1:5000/
```

## API Endpoints

### 1. Home Endpoint

Method: GET

```text
/
```

Checks whether the Flask application is running.

### 2. Health Check

Method: GET

```text
/health
```

Checks API availability and model loading status.

### 3. House Price Prediction

Method: POST

```text
/predict
```

Accepts house features in JSON format.

Example request:

```json
{
  "features": {
    "actual_feature_name_1": 100,
    "actual_feature_name_2": 3
  }
}
```

Replace the example feature names and values with the actual features expected by your trained model.

Example response:

```json
{
  "status": "success",
  "predicted_price": 250000.0
}
```

*Note: The predicted price shown above is illustrative.*

## Model Deployment

The trained Machine Learning model is loaded using Joblib.

Flask handles incoming API requests, validates the input data, passes the features to the trained model, and returns the prediction.

## Testing

The API can be tested using:

- Jupyter Notebook
- Python Requests library
- CURL commands

## Conclusion

This project demonstrates how a trained Machine Learning model can be deployed as a Flask REST API.

It provides a simple way to integrate Machine Learning predictions into applications through HTTP requests and JSON responses.
