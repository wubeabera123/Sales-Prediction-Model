from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Define the FastAPI app
app = FastAPI()

# Define the input data model for predictions
class SalesPredictionInput(BaseModel):
    Store: int
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: int
    Promo2SinceWeek: float
    Promo2SinceYear: float
    DayOfWeek: int
    MonthDay: int
    IsWeekend: int
    IsBeginningOfMonth: int
    IsMidMonth: int
    IsEndOfMonth: int
    PromoInterval: str
    Assortment_b: bool
    Assortment_c: bool
    StateHoliday_a: bool
    StateHoliday_b: bool
    StateHoliday_c: bool

# Load the trained model
model = joblib.load("saved_models/model-22-09-2024-14-59-02-071.pkl")  # Update with your correct path to the model

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict_sales(input_data: SalesPredictionInput):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Preprocess the input data if necessary (depends on your original pipeline)
    # Example: input_df = preprocess_input(input_df)

    # Predict using the model
    prediction = model.predict(input_df)

    # Return the prediction as a JSON response
    return {"predicted_sales": prediction[0]}
