from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import cv2
from doctr.models import ocr_predictor
from deep_translator import GoogleTranslator
from pathlib import Path

app = FastAPI(title="Comic Translate Web")

# initialize OCR model once
ocr_model = ocr_predictor(det_arch='db_resnet34', reco_arch='parseq', pretrained=True)

template = Path(__file__).with_name("templates").joinpath("index.html")

@app.get("/", response_class=HTMLResponse)
async def index():
    return template.read_text()

@app.post("/translate")
async def translate_image(target_lang: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = ocr_model([img])

    texts = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                if line_text.strip():
                    texts.append(line_text)

    translator = GoogleTranslator(source="auto", target=target_lang)
    translations = [translator.translate(t) for t in texts]

    return JSONResponse({"texts": texts, "translations": translations})
