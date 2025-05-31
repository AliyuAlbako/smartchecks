from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
dataset = load_from_disk('./dataset/hausa_fake_news_dataset')

# Load model and tokenizer
model_dir = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare test data
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# Predict on test set
predictions = []

for text in tqdm(test_texts):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_class = torch.argmax(logits, dim=-1).cpu().item()
    predictions.append(pred_class)

# Evaluate results
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions, target_names=["Real News", "Fake News"])

# Print results
print(f"\nAccuracy on Hausa Fake News Test Set: {accuracy:.4f}\n")
print("Detailed classification report:\n")
print(report)
