from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
from torchvision import models, transforms
import torch
import io
import os
import timm

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the CGM Image & Chat API. Visit /docs to test."}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 🤖 Initialize OpenAI Client
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# 🔬 Initialize Image Classifier
# ----------------------------

cv_model = timm.create_model("resnet50d", pretrained=True)
cv_model.eval()

# Food-101 labels (loaded from timm's metadata)
food101_labels = cv_model.pretrained_cfg["label_names"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cv_model.pretrained_cfg["mean"], std=cv_model.pretrained_cfg["std"]),
])

# ------------------------
# 🗨️ /chat endpoint
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

# ------------------------
# 🖼️ /analyze image endpoint
# ------------------------

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
            "label": food101_labels[class_id.item()],
            "confidence": float(confidence)
        }

    except Exception as e:
        return {"error": str(e)}
