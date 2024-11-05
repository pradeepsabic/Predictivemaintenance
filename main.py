# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import extract_features
from src.modeling import train_model, PredictiveModel

app = FastAPI()

# Load and preprocess data, train model here
maintenance_logs, sensor_data = load_data('data/logs.csv', 'data/sensors.csv')
maintenance_logs, sensor_data = clean_data(maintenance_logs, sensor_data)
features = extract_features(maintenance_logs, sensor_data)

input_size = features.shape[1]  # Number of features
target = pd.Series(...)  # Define your target variable here
model = train_model(features, target, input_size)

# Define Pydantic model for request body
class PredictionRequest(BaseModel):
    # Define your feature fields here. For example:
    feature1: float
    feature2: float
    # Add other features as required

@app.post('/predict')
async def predict(request: PredictionRequest):
    # Convert request data to DataFrame or appropriate format
    data = request.dict()
    # Assuming extract_features can handle a dict, you may need to convert to DataFrame
    features = extract_features(pd.DataFrame([data]))  # Wrap in list to create DataFrame
    prediction = model.predict(features)

    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
