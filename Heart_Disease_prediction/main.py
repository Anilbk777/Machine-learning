# main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import pandas as pd
import pickle


with open("model/heart_disease.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Heart Disease Prediction API")


class PatientData(BaseModel):
    Age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the patient in years (1-119)", example=55)]
    Sex: Annotated[Literal['M', 'F'], Field(..., description="Sex of the patient: M = Male, F = Female", example="M")]
    ChestPainType: Annotated[Literal['ATA', 'NAP', 'ASY', 'TA'], Field(..., description="Chest pain type", example="ATA")]
    RestingBP: Annotated[int, Field(..., gt=50, lt=250, description="Resting blood pressure in mm Hg (50-250)", example=140)] 
    Cholesterol: Annotated[int, Field(..., gt=100, lt=600, description="Serum cholesterol in mg/dl (101-599)", example=220)] 
    FastingBS: Annotated[int, Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0 or 1)", example=0)] 
    RestingECG: Annotated[Literal['Normal', 'ST', 'LVH'], Field(..., description="Resting ECG result", example="Normal")]
    MaxHR: Annotated[int, Field(..., gt=60, lt=250, description="Maximum heart rate achieved (61-249)", example=150)] 
    ExerciseAngina: Annotated[Literal['N', 'Y'], Field(..., description="Exercise induced angina", example="N")] 
    Oldpeak: Annotated[float, Field(..., ge=0.0, lt=10.0, description="ST depression induced by exercise (0-10 mm)", example=2.3)] 
    ST_Slope: Annotated[Literal['Up', 'Flat', 'Down'], Field(..., description="Slope of peak exercise ST segment", example="Up")]


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}
from fastapi.responses import JSONResponse

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([{
        'Age': data.Age,
        'Sex': data.Sex,
        'ChestPainType': data.ChestPainType,
        'RestingBP': data.RestingBP,
        'Cholesterol': data.Cholesterol,
        'FastingBS': data.FastingBS,
        'RestingECG': data.RestingECG,
        'MaxHR': data.MaxHR,
        'ExerciseAngina': data.ExerciseAngina,
        'Oldpeak': data.Oldpeak,
        'ST_Slope': data.ST_Slope
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  

    return JSONResponse(
        status_code=200,
        content={
            "HeartDiseasePrediction": int(prediction),
            "Probability": round(float(probability), 4)
        }
    )



import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
