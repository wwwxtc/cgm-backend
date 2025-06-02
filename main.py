from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MealData(BaseModel):
    description: str
    carbs: float
    protein: float
    fat: float

@app.post("/predict")
def predict(data: MealData):
    predicted_cgm = 180 + data.carbs * 0.5 - data.fat * 0.3
    advice = f"For your meal '{data.description}', predicted CGM is {predicted_cgm:.1f} mg/dL."
    return {"prediction": predicted_cgm, "advice": advice}
