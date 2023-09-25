from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline, __version__ as model_version

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str

@app.get("/")
def home():
    """
    Endpoint to perform a health check and provide the model version.
    """
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    """
    Endpoint to predict the language based on the input text.
    """
    language = predict_pipeline(payload.text)
    return {"language": language}
