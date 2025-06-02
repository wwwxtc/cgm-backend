from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the CGM Prediction API! Visit /docs to test."}

# Enable CORS for development (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client using SDK v1.x
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# 📌 /predict endpoint section
# ----------------------------

class MealData(BaseModel):
    description: str
    carbs: float
    protein: float
    fat: float

def get_chatgpt_advice(meal_desc: str, predicted_cgm: float) -> str:
    prompt = (
        f"My glucose is predicted to rise to {predicted_cgm:.1f} mg/dL after eating: {meal_desc}. "
        f"What advice would you give?"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful diabetes assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(LLM Error) {str(e)}"

@app.post("/predict")
def predict(data: MealData):
    predicted_cgm = 180 + data.carbs * 0.5 - data.fat * 0.3
    advice = get_chatgpt_advice(data.description, predicted_cgm)
    return {
        "prediction": predicted_cgm,
        "advice": advice
    }

# ------------------------
# 🗨️ /chat endpoint section
# ------------------------

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(data: ChatInput):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful diabetes assistant."},
                {"role": "user", "content": data.message}
            ],
            temperature=0.7
        )
        return {
            "reply": response.choices[0].message.content
        }
    except Exception as e:
        return {"error": str(e)}
