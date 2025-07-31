from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import requests
import io

app = FastAPI()

# Enable CORS (for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Download model from Google Drive
# -----------------------------
MODEL_FILE_ID = "1AbCDeFgHiJkLmNopQRsTu"  # ← replace this with your file ID
MODEL_PATH = "efficientnet_b7_food_classifier.h5"

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

download_model_from_drive()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels and calories (example)
class_labels = ["Apple", "Banana", "Burger", "Pizza", "Rice", "Salad"]
calories = {
    "Apple": 52,
    "Banana": 89,
    "Burger": 295,
    "Pizza": 266,
    "Rice": 130,
    "Salad": 33
}

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # adjust for your model
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        predicted_calories = calories.get(predicted_class, "Unknown")

        return {
            "predicted_food": predicted_class,
            "calories": predicted_calories,
            "confidence": float(prediction[predicted_index])
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional health check
@app.get("/")
def root():
    return {"message": "Food recognition API is running"}

