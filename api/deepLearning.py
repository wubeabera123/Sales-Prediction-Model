from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

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

# Load the saved model (using .h5 format)
model = load_model("saved_models/model-23-09-2024-21-34-11-828.h5")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API using deep learning"}

# Prediction endpoint
@app.post("/predict/")
def predict_sales(input_data: SalesPredictionInput):
    # Convert the input data to a numpy array (adjust this to match your model's input)
    input_array = np.array([[input_data.Store,
                             input_data.CompetitionDistance,
                             input_data.CompetitionOpenSinceMonth,
                             input_data.CompetitionOpenSinceYear,
                             input_data.Promo2,
                             input_data.Promo2SinceWeek,
                             input_data.Promo2SinceYear,
                             input_data.DayOfWeek,
                             input_data.MonthDay,
                             input_data.IsWeekend,
                             input_data.IsBeginningOfMonth,
                             input_data.IsMidMonth,
                             input_data.IsEndOfMonth,
                             input_data.Assortment_b,
                             input_data.Assortment_c,
                             input_data.StateHoliday_a,
                             input_data.StateHoliday_b,
                             input_data.StateHoliday_c]])

    # Predict using the loaded model
    prediction = model.predict(input_array)

    # Convert the prediction to a native Python data type (e.g., float)
    predicted_sales = float(prediction[0][0])

    # Return the prediction as a JSON response
    return {"predicted_sales": predicted_sales}
