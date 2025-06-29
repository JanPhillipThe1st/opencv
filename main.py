from fastapi import FastAPI, File, UploadFile, Body, status
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
import io
from pydantic import BaseModel, model_validator
import json

class ThreshSettings(BaseModel):
    single_channel: bool
    threshold: float = 177
    max_value: float = 255
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
    
class MorphSettings(BaseModel):
    single_channel: bool
    threshold: float = 177
    max_value: float = 255
    morph_kernel_x: int = 7
    morph_kernel_y: int = 7
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
    
app = FastAPI()
@app.post("/thresh-image/")
async def process_image(settings:ThreshSettings=Body(...),file:UploadFile = File(...)):
    contents = await file.read()
    settings.model_dump()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # === Your OpenCV Logic Here ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if settings.single_channel:
        ret, thresh = cv2.threshold(gray,settings.threshold,settings.max_value,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        ret, thresh = cv2.threshold(img,settings.threshold,settings.max_value,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        

    # Encode back to image bytes
    _, im_buf_arr = cv2.imencode(".jpg", thresh)
    io_buf = io.BytesIO(im_buf_arr.tobytes())

    return StreamingResponse(io_buf, media_type="image/jpeg")

@app.post("/gray-image/")
async def gray_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # === Your OpenCV Logic Here ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encode back to image bytes
    _, im_buf_arr = cv2.imencode(".jpg", gray)
    io_buf = io.BytesIO(im_buf_arr.tobytes())

    return StreamingResponse(io_buf, media_type="image/jpeg")

@app.post("/image-morph/")
async def gray_image(settings:MorphSettings=Body(...),file: UploadFile = File(...)):
    contents = await file.read()
    settings.model_dump()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # === Your OpenCV Logic Here ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray,settings.threshold,settings.max_value,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # apply morphology
    kernel = np.ones((settings.morph_kernel_x,settings.morph_kernel_y), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)


    # Encode back to image bytes
    _, im_buf_arr = cv2.imencode(".jpg", morph)
    io_buf = io.BytesIO(im_buf_arr.tobytes())

    return StreamingResponse(io_buf, media_type="image/jpeg")