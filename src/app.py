from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

class PredictionInput(BaseModel):
    TenureMonths: int
    ContractType: str
    SupportTickets: int
    MonthlyCharge: float

app = FastAPI()

# Load the trained model
model = joblib.load('models/churn_model.pkl')

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running. Use the /predict endpoint for predictions."}

@app.post("/predict/")
def predict_churn(input_data: PredictionInput):
    """
    Predicts churn based on input data.
    """
    # Convert input to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return {
        "prediction": int(prediction[0]),
        "churn_probability": float(probability[0][1])
    }
