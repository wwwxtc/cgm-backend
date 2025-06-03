from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
from torchvision import models, transforms
import torch
import io
import os

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the CGM Image & Chat API. Visit /docs to test."}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Load MobileNetV2
cv_model = models.mobilenet_v2(pretrained=True)
cv_model.eval()

# ImageNet labels
imagenet_labels = [f"class_{i}" for i in range(1000)]
imagenet_labels[954] = "banana"

# Transform for classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Chat input schema
class ChatInput(BaseModel):
    message: str

# 🔹 GPT-4o text chat endpoint
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

# 🔹 MobileNet image classifier endpoint
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = cv_model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_id = torch.max(probs, 0)

        return {
            "label": imagenet_labels[class_id.item()],
            "confidence": float(confidence)
        }
    except Exception as e:
        return {"error": str(e)}

# 🔹 Gemini Vision + Text endpoint
@app.post("/visionchat")
async def vision_chat(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        contents = await file.read()
        image_data = Image.open(io.BytesIO(contents)).convert("RGB")
        image_bytes = io.BytesIO()
        image_data.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        response = gemini_model.generate_content(
            [genai.types.Part.from_data(data=image_bytes.read(), mime_type="image/jpeg"), prompt],
            stream=False
        )

        return {"reply": response.text}
    except Exception as e:
        return {"error": str(e)}
