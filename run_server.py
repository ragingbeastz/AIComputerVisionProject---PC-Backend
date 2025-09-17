from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64
import uvicorn

import temp_image_processing as temp_imageprocess_image



API_KEY = "abcdefg"

backend = FastAPI()

@backend.post("/process")
async def identifyCar(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read image
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    #Process Image
    img = temp_imageprocess_image.process_image(img)
    img = Image.fromarray(img)

    # Encode image to base64 for JSON
    out_buf = BytesIO()
    img.save(out_buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(out_buf.getvalue()).decode()

    #Get Text
    car_make = "Ford Fiesta 2010"  # placeholder
    car_info = """The 2010 Ford Fiesta is a popular choice in the used car market, particularly known for its fun-to-drive nature and strong safety ratings. 
Engines: Available with a range of petrol and diesel engines, including 1.25L, 1.4L, and 1.6L petrol options and 1.4L and 1.6L TDCi diesels.
Driving Dynamics: Praised for its precise steering and agile chassis, making it enjoyable to drive. The suspension effectively handles bumps, providing a comfortable ride.
Safety: Received a five-star Euro NCAP crash test rating. Standard safety features include dual front, side and driver's knee airbags, seatbelt pre-tensioners, all-round three-point seatbelts, and ISOFIX child-seat mounting points. Optional safety features like Emergency City Stop and Ford's MyKey system, which allows parents to set speed and volume limits for young drivers, are also available.
Interior Features: The Mk6 Fiesta features Ford's Convers+ menu system and steering wheel controls. Other features include keyless entry (with a "Ford Power" starter button), adjustable steering wheel, electric power steering, and a USB port. """

    return JSONResponse({
        "car_make": car_make,
        "car_info": car_info,
        "image_base64": f"data:image/jpeg;base64,{b64_img}"
    })


if __name__ == "__main__":
    uvicorn.run(backend, host="0.0.0.0", port=8000)