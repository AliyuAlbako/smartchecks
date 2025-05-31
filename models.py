
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
import torch
logging.set_verbosity_error()


fine_tuned_model =  "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model, num_labels=2)

def predict_input(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    label = "Misinformation" if prediction.item() == 1 else "Verified"

    return {
        "text": text,
        "classification": label,
        "confidence_score": float(confidence.item()),
           }

