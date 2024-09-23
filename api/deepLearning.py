from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Define the FastAPI app
app = FastAPI()

# Define the input data model for predictions (adjust these fields based on your use case)
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

# Load the saved model architecture and weights from .pkl files
with open("saved_models/model_architecture-23-09-2024-17-28-04-042.pkl", "rb") as f:
    model = pickle.load(f)

with open("saved_models/model_weights-23-09-2024-17-28-04-042.pkl", "rb") as f:
    weights = pickle.load(f)

# Set the loaded weights to the model (Keras models usually expect set_weights())
model.set_weights(weights)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict_sales(input_data: SalesPredictionInput):
    # Convert the input data to a numpy array (adjust the order to match your model's input)
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

    # Reshape the input to match the model's expected input shape if necessary
    # (e.g., model expects 3D input for LSTM or 2D for Dense layers)
    # For LSTM, it might look like this:
    # input_array = input_array.reshape((1, input_array.shape[1], 1))

    # Predict using the model
    prediction = model.predict(input_array)

    # Return the prediction as a JSON response
    return {"predicted_sales": prediction[0][0]}
