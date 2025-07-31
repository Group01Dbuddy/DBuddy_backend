# Load JSON with food -> calories
with open("food_data.json", "r") as f:
    food_data = json.load(f)

class_labels = list(food_data.keys())

...

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        predicted_calories = food_data.get(predicted_class, "Unknown")

        return {
            "predicted_food": predicted_class,
            "calories": predicted_calories,
            "confidence": float(prediction[predicted_index])
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
