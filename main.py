from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from img_processor import process_image

import os
import shutil

app = FastAPI()

#static folder for images
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#check for static directory + create if does not exist
os.makedirs("static", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    #file paths
    input_path = f"static/original_{file.filename}"
    output_path = f"static/processed_{file.filename}"

    #uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_image(input_path, output_path)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "original": input_path,
            "processed": output_path
        }
    )