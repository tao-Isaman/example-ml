# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import os

app = FastAPI()

# Load model
model_path = "model/iris_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found!")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Mapping for Iris target names
iris_target_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Input format
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction API
@app.post("/predict")
def predict(data: IrisRequest):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction_index = model.predict(input_data)[0]
    prediction_name = iris_target_names[prediction_index]

    # Monitoring (save predictions)
    with open("monitor.log", "a") as log_file:
        log_file.write(f"Predicted: {prediction_name}, Input: {input_data.tolist()}\n")

    return {
        "prediction": prediction_name
    }

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
