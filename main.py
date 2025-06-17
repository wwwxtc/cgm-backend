import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import logging
from typing import List
import numpy as np
import threading
import tflite_runtime.interpreter as tflite



logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more detail if needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
chat_history = []  # In-memory history for context-aware chat


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

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Welcome route
@app.get("/")
def read_root():
    return {"message": "Welcome to the CGM Image & Chat API. Visit /docs to test."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/chat-history")
def get_chat_history():
    return {"history": chat_history}

@app.post("/clear-chat-history")
def clear_history():
    chat_history.clear()
    return {"message": "Chat history cleared."}


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Chat input schema
class ChatInput(BaseModel):
    message: str

# Chat endpoint using OpenAI GPT-4o
@app.post("/chat")
@limiter.limit("10/minute")  # Step 3: Apply rate limit here
async def chat(request: Request, data: ChatInput):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
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
        logger.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Gemini vision endpoint
@app.post("/analyze-image/")
@limiter.limit("5/minute")  # Step 3: Apply rate limit here
async def analyze_image(request: Request, image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"]
        if image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG, PNG, or WEBP.")

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
        logger.error(f"Error in /analyze-image/: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ChatGPT_4o contextual endpoint
@app.post("/chat-contextual")
@limiter.limit("10/minute")
async def chat_contextual(request: Request, data: ChatInput):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Limit to last 10 messages to avoid token overflow
    trimmed_history = chat_history[-10:]

    messages = (
        [{"role": "system", "content": "You are a helpful diabetes assistant."}]
        + trimmed_history
        + [{"role": "user", "content": data.message}]
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content

        # Save to history
        chat_history.append({"role": "user", "content": data.message})
        chat_history.append({"role": "assistant", "content": reply})

        return {"reply": reply}
    except Exception as e:
        logger.error(f"Error in /chat-contextual: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

# Load TFLite model
interpreter = tflite.Interpreter(model_path="glunet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter_lock = threading.Lock()

@app.post("/predict")
def predict(input_data: List[float]):
    with interpreter_lock:
        try:
            input_array = np.array(input_data, dtype=np.float32).reshape(input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            return {"prediction": output.tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
