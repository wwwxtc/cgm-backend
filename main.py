import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Welcome route
@app.get("/")
def read_root():
    return {"message": "Welcome to the CGM Image & Chat API. Visit /docs to test."}

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Chat input schema
class ChatInput(BaseModel):
    message: str

# Chat endpoint using OpenAI GPT-4o
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
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Gemini vision endpoint
@app.post("/analyze-image/")
async def analyze_image(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"]
        if image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG, PNG, or WEBP.")

    
        # Gemini accepts binary image content directly with the prompt
        response = model.generate_content(
            [
                {
                    "mime_type": image.content_type,
                    "data": image_bytes
                },
                "Describe this image"
            ]
        )

        return {"text": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
