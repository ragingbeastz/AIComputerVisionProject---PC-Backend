from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input, decode_predictions
import temp_image_processing as temp_imageprocess_image
from transformers import pipeline
import requests



with open("auth.txt", "r") as f:
    lines = f.read().splitlines()
    google_API_KEY = lines[0]
    google_CX = lines[1]

googleSearch_url = "https://www.googleapis.com/customsearch/v1"


API_KEY = "abcdefg"
backend = FastAPI()

@backend.post("/process")
async def identifyCar(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read image
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Pre-process Image
    img = temp_imageprocess_image.process_image(img)
    img = Image.fromarray(img)

    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)   


    # Load Models and Predict

    #Car Colour
    colour_model = tf.keras.models.load_model(r"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\model_colours.keras")
    colour_labels = pd.read_csv(r"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\labels_colours.csv")

    preds_colour = colour_model.predict(x)
    top_colour = tf.keras.applications.efficientnet.decode_predictions(preds_colour, top=1)[0][0]
    car_colour = colour_labels.iloc[top_colour[0]]['label']

    # Car Make
    make_model = tf.keras.models.load_model(r"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\model_car_make.keras")
    make_labels = pd.read_csv(r"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\labels_car_make.csv")

    preds_make = make_model.predict(x)
    top_make = tf.keras.applications.efficientnet.decode_predictions(preds_make, top=1)[0][0]
    car_make = make_labels.iloc[top_make[0]]['label']

    # Car Model
    model_model = tf.keras.models.load_model(rf"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\model_car_model_{car_make}.keras")
    model_labels = pd.read_csv(rf"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\labels_car_model_{car_make}.csv")

    preds_model = model_model.predict(x)
    top_model = tf.keras.applications.efficientnet.decode_predictions(preds_model, top=1)[0][0]
    car_model = model_labels.iloc[top_model[0]]['label']

    # Car Year
    year_model = tf.keras.models.load_model(rf"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\model_car_year_{car_make}_{car_model}.keras")  
    year_labels = pd.read_csv(rf"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\labels_car_year_{car_make}_{car_model}.csv")

    preds_year = year_model.predict(x)
    top_year = tf.keras.applications.efficientnet.decode_predictions(preds_year, top=1)[0][0]
    car_year = year_labels.iloc[top_year[0]]['label']

    result = f"{car_colour} {car_make} {car_model} {car_year}"

    # Encode image to base64 for JSON
    out_buf = BytesIO()
    img.save(out_buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(out_buf.getvalue()).decode()

    #Get Text
    generator = pipeline("text-generation", model="google/gemma-2-2b-it")
    prompt = f"Write a detailed description of the {car_year} {car_make} {car_model}, including key features, safety, and driving experience."
    out = generator(
        prompt,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        return_full_text=False
    )
    car_info = out[0]["generated_text"]

    #Get Image
    params = {
        "q": f"{car_colour} {car_year} {car_make} {car_model}",
        "cx": google_CX,
        "key": google_API_KEY,
        "searchType": "image",
        "num": 1, 
    }
    response = requests.get(googleSearch_url, params=params).json()
    first_image_url = response["items"][0]["link"]

    google_image_base64 = None
    if first_image_url:
        img_response = requests.get(first_image_url)
        if img_response.status_code == 200:
            google_image_base64 = base64.b64encode(img_response.content).decode()


    return JSONResponse({
        "result": result,
        "car_info": car_info,
        "image_base64": f"data:image/jpeg;base64,{google_image_base64}" if google_image_base64 else None
    })


if __name__ == "__main__":
    uvicorn.run(backend, host="0.0.0.0", port=8000)