import pickle
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load the trained model
model_filename = f"ml-project-ue-katowice.pkl"
model_path = BASE_DIR / model_filename

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define the classes
classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

def predict_pipeline(text):
    # Preprocess the text
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = text.lower()

    # Predict the class for the input text
    pred = model.predict([text])
    return classes[pred[0]]
