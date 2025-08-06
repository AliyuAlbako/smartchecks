

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# from models import predict_input
from models import predict_news, vectorizer

import uvicorn
import json
from datetime import datetime
from typing import Optional

from utils.text_cleaning import clean_text

app = FastAPI()



app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class FeedbackRequest(BaseModel):
    text: str
    ai_classification: str
    user_feedback: str
    reason: Optional[str] = None
class NewsRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# @app.post("/check")
# def check_misinformation(request: TextRequest):
#     result= predict_input(request.text)
#
#     return {"text": request.text,
#            "classification": result["classification"],
#             "confidence":result["confidence_score"]
#             }

@app.post("/check")
def check_misinformation(request: NewsRequest):
    result= predict_news(request.text)

    return {
        "text": request.text,
        "classification": result.prediction,
        "confidence":result.confidence


    }







@app.get("/feedback-form", response_class=HTMLResponse)
def feedback_form(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})

@app.post("/feedback")
def submit_feedback( text: str = Form(...),
    ai_classification: str = Form(...),
    user_feedback: str = Form(...),
    reason: str = Form(...)
):
    feedback_entry = {
        "text": text,
        "ai_classification": ai_classification,
        "user_feedback": user_feedback,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Load existing feedbacks
    try:
        with open("feedback_store.json", "r") as file:
            feedback_data = json.load(file)
    except FileNotFoundError:
        feedback_data = []

    # Append new feedback
    feedback_data.append(feedback_entry)

    # Save updated feedbacks
    with open("feedback_store.json", "w") as file:
        json.dump(feedback_data, file, indent=4)

    return {"message": "Feedback submitted successfully", "feedback": feedback_entry}

@app.get("/feedbacks")
def get_feedbacks():
    try:
        with open("feedback_store.json", "r") as file:
            feedback_data = json.load(file)
    except FileNotFoundError:
        feedback_data = []

    return {"feedbacks": feedback_data}

@app.get("/dashboard", response_class=HTMLResponse)
def view_dashboard(request: Request):
    try:
        with open("feedback_store.json", "r") as file:
            feedback_data = json.load(file)
    except FileNotFoundError:
        feedback_data = []

    total_feedback = len(feedback_data)
    total_corrections = sum(1 for fb in feedback_data if fb["ai_classification"] != fb["user_feedback"])

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "feedbacks": feedback_data,
        "total_feedback": total_feedback,
        "total_corrections": total_corrections
    })
@app.post("/predict_feed", response_class=HTMLResponse)
async def predict_from_feed(request: Request, text: str = Form(...)):
    prediction = predict_news(text)
    return templates.TemplateResponse("result_from_feed.html", {"request": request, "text": text, "prediction": prediction})


@app.get("/feed", response_class=HTMLResponse)
async def show_feed(request: Request):
    # Load sample texts
    sample_posts = [
        "Boko Haram attack leaves 30 dead in Maiduguri.",
        "Government declares free food for every citizen.",
        "Fake news: No bombing occurred in Kaduna today.",
        "Peace treaty signed in Southern Nasarawa."
    ]
    return templates.TemplateResponse("feed.html", {"request": request, "posts": sample_posts})



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)