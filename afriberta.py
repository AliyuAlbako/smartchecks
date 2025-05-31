from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Davlan/afro-xlmr-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2 )

print("Successfully loaded Afro-XLMR model ✅")
