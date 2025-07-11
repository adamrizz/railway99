import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "daun_padi_cnn_model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
CLASS_NAMES = ["Bacterial Leaf Blight", "Leaf Blast", "Leaf Scald", "Brown Spot", "Narrow Brown Spot", "Healthy"]

@app.get("/")
def home():
    return {"message": "API klasifikasi daun padi siap"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    contents = await file.read()
    _, height, width, _ = input_details[0]['shape']
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((width, height))
    img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    score = predictions[0]
    predicted_index = int(np.argmax(score))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(score))

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "all_predictions": {CLASS_NAMES[i]: float(score[i]) for i in range(len(CLASS_NAMES))}
    }
