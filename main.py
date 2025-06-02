from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 🔓 Allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all domains — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
