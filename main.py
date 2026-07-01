from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from img_processor import process_image

import os
import shutil

app = FastAPI()

STATIC_DIR = "static"

#static folder for images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")

#check for static directory + create if does not exist
os.makedirs(STATIC_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

MODELS = {
    "convnext": "ConvNeXt model",
    "mednext_kan": "MedNeXt KAN model",
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"models": MODELS}
    )

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request, model: str):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model": model, "model_label": MODELS.get(model, model)}
    )

@app.post("/upload/")
async def upload_images(request: Request, model: str = Form(...), files: list[UploadFile] = File(...)):
    #clear frames from any previous upload
    for name in os.listdir(STATIC_DIR):
        os.remove(os.path.join(STATIC_DIR, name))

    #folder pickers include non-image files (e.g. .DS_Store) - skip those
    images = [f for f in files if os.path.splitext(f.filename)[1].lower() in IMAGE_EXTENSIONS]
    #preserve folder order so the "video" plays back in the right sequence
    images.sort(key=lambda f: f.filename)

    frames = []
    for i, file in enumerate(images):
        input_path = f"{STATIC_DIR}/original_{i:03}{os.path.splitext(file.filename)[1]}"
        output_path = f"{STATIC_DIR}/frame_{i:03}.png"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        process_image(input_path, output_path)
        frames.append(output_path)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model": model,
            "model_label": MODELS.get(model, model),
            "frames": frames
        }
    )