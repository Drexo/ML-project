from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline

app = FastAPI()

class TextIn(BaseModel):
    text: str
    text2: str
    text3: str

class PredictionOut(BaseModel):
    language: str
    language2: str
    language3: str

@app.get("/")
def home():
    """
    Endpoint to perform a health check.
    """
    return {"API": "OK"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    """
    Endpoint to predict the language based on the input text.
    """
    language = predict_pipeline(payload.text)
    language2 = predict_pipeline(payload.text2)
    language3 = predict_pipeline(payload.text3)
    return PredictionOut(language=language, language2=language2, language3=language3)
