from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load trained model
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example text
text = "Breaking: Aliens landed at the president's residence."

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probs = torch.softmax(logits, dim=1)
prediction = torch.argmax(probs, dim=1).item()

label_map = {0: "True News", 1: "Fake News"}
print(f"Prediction: {label_map[prediction]} (Confidence: {probs.max().item():.4f})")
