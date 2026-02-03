from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# app.mount("/static", StaticFiles(directory="static"), name="static")

from pydantic import BaseModel
import uvicorn
import sys

from hate.pipeline.train_pipeline import TrainPipeline
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import *

app = FastAPI(title="Hate Speech Detection App")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )


@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return {"message": "Training completed successfully"}

    except Exception as e:
        return {"error": str(e)}



@app.post("/predict")
async def predict_api(request: PredictRequest):
    try:
        pipeline = PredictionPipeline()
        result = pipeline.run_pipeline(request.text)
        return {
            "input": request.text,
            "prediction": result
        }

    except Exception as e:
        raise CustomException(e, sys)


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request):
    try:
        form = await request.form()
        text = form.get("text")

        pipeline = PredictionPipeline()
        result = pipeline.run_pipeline(text)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "text": text
            }
        )

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
