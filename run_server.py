import cv2
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
import image_processing as temp_imageprocess_image
from transformers import pipeline
import requests


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CarVision/1.0)"})

with open("auth.txt", "r") as f:
    lines = f.read().splitlines()
    google_API_KEY = lines[0]
    google_CX = lines[1]

googleSearch_url = "https://www.googleapis.com/customsearch/v1"


colour_model_path = ""
colour_labels_path = ""
make_model_path = ""
make_labels_path = ""
model_model_path = ""
model_labels_path = ""
year_model_path = ""
year_labels_path = ""


API_KEY = "abcdefg"
backend = FastAPI()

@backend.post("/process")
async def identifyCar(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read image
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.rotate(-90, expand=True)

    save_path = "uploaded_image.jpg"
    img.save(save_path)

    # Pre-process Image
    img = temp_imageprocess_image.process_image(img)
    processed_path = "processed_image.jpg"
    cv2.imwrite(processed_path, img)
    img = Image.fromarray(img)

    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)   


    print("Image pre-processed")
    print("Predicting...")

    # Load Models and Predict

    try:
        #Car Colour
        colour_model = tf.keras.models.load_model(colour_model_path, compile=False)
        colour_labels = pd.read_csv(colour_labels_path)

        preds_colour = colour_model.predict(x)
        idx_colour = int(np.argmax(preds_colour, axis=1)[0])
        car_colour = colour_labels.iloc[idx_colour]['label']

    except Exception as e:
        print(f"Colour model not found. Setting colour as 'Unknown'. Error: {e}")
        car_colour = ""

    try:
        # Car Make
        make_model = tf.keras.models.load_model(make_model_path, compile=False)
        make_labels = pd.read_csv(make_labels_path)

        preds_make = make_model.predict(x)
        idx_make = int(np.argmax(preds_make, axis=1)[0])
        car_make = make_labels.iloc[idx_make]['label']
    
    except Exception as e:
        print(f"Make model not found. Setting make as 'Unknown'. Error: {e}")
        car_make = ""
    
    try:
        # Car Model
        model_model = tf.keras.models.load_model(model_model_path, compile=False)
        model_labels = pd.read_csv(model_labels_path)

        preds_model = model_model.predict(x)
        #Print top 5
        print("Top 5 Model Predictions:")
        top_5_indices = np.argsort(preds_model[0])[-5:][::-1]
        for i in top_5_indices:
            print(f"{model_labels.iloc[i]['label']}: {preds_model[0][i]:.4f}")
        idx_model = int(np.argmax(preds_model, axis=1)[0])
        
        car_model = model_labels.iloc[idx_model]['label']
    
    except Exception as e:
        print(f"Model model not found for {car_make}. Setting model as 'Unknown'. Error: {e}")
        car_model = ""

    # Car Year
    try:
        year_model = tf.keras.models.load_model(year_model_path, compile=False)
        year_labels = pd.read_csv(year_labels_path)

        preds_year = year_model.predict(x)
        idx_year = int(np.argmax(preds_year, axis=1)[0])
        car_year = year_labels.iloc[idx_year]['label']
    
    except Exception as e:
        print(f"Year model not found for {car_make} {car_model}. Setting year as 'Unknown'. Error: {e}")
        car_year = ""

    result = f"{car_colour} {car_make} {car_model} {car_year}"

    print("Predicted:", result)

    # Encode image to base64 for JSON
    out_buf = BytesIO()
    img.save(out_buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(out_buf.getvalue()).decode()

    #Get Text
    print("Generating car description...")
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
    print("Description generated:", car_info)

    print("Fetching car image from Google Custom Search...")
    #Get Image
    params = {
        "q": f"{car_colour} {car_year} {car_make} {car_model}",
        "cx": google_CX,
        "key": google_API_KEY,
        "searchType": "image",
        "num": 1, 
    }
    response = SESSION.get(googleSearch_url, params=params).json()
    first_image_url = response["items"][0]["link"]
    img_response = SESSION.get(first_image_url)
    google_image_base64 = base64.b64encode(img_response.content).decode()

    print("Fetched image from Google Custom Search.")
    
    print("REQUEST COMPLETE")
    return JSONResponse({
        "result": result,
        "car_info": car_info,
        "image_base64": f"data:image/jpeg;base64,{google_image_base64}" 
    })


if __name__ == "__main__":
    uvicorn.run(backend, host="0.0.0.0", port=8000)