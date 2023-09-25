import pickle
import re

# Load the trained model
model_path = f"ml-project-ue-katowice.pkl"

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
    "Portuguese",
    "Russian",
    "Spanish",
    "Swedish",
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
