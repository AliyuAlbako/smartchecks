#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
# import torch
# logging.set_verbosity_error()
#
#
# # fine_tuned_model =  "./fine_tuned_model"
# model_name = "prajjwal1/bert-tiny"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
#
# def predict_input(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     probs = torch.softmax(outputs.logits, dim=1)
#     confidence, prediction = torch.max(probs, dim=1)
#
#     label = "Fake News" if prediction.item() == 1 else "Real News"
#
#     return {
#         "text": text,
#         "classification": label,
#         "confidence_score": float(confidence.item()),
#            }

# second trial
# import joblib
#
# # Load the lightweight model
#
# model = joblib.load("lightmodel/fake_news_model.pkl")
#
# def predict_input(text):
#     prediction = model.predict([text])[0]
#     probabilities = model.predict_proba([text])[0]
#     confidence = max(probabilities)
#
#     label = "Real News" if prediction == "true" else "Fake News"
#
#     return {
#         "text": text,
#         "classification": label,
#         "confidence_score": float(confidence),
#     }

# third trial

# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

from utils.text_cleaning import clean_text

# Load model and vectorizer at startup
MODEL_PATH = "lightmodel/fake_news_model.pkl"
VECTORIZER_PATH = "lightmodel/tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# Pydantic schema

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


# @app.post("/predict", response_model=PredictionResponse)
def predict_news(text):
    try:
        cleaned = clean_text(text)
        transformed = vectorizer.transform([cleaned])
        pred_proba = model.predict_proba(transformed)[0]
        pred_class = model.predict(transformed)[0]

        return PredictionResponse(
            prediction="Fake" if pred_class == 1 else "Real",
            confidence=round(max(pred_proba), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
