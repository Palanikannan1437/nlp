from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import summarize
from app.model.model import predict_personality_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()
class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    personality: str


class SummarizeOut(BaseModel):
    summary: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict-personality", response_model=PredictionOut)
def predict(payload: TextIn):
    personality = predict_personality_pipeline(payload.text)
    return {"personality": personality}

@app.post("/summarize", response_model=SummarizeOut)
def predict(payload: TextIn):
    summary = summarize(payload.text,3)
    return {"summary": summary}