from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import torch

# Load the large dataset
fake_df = pd.read_csv("dataset/fake.csv")
true_df = pd.read_csv("dataset/true.csv")
#triming the dataset
fake_df_small= fake_df.head(500)
true_df_small= true_df.head(500)

# saving the trimmed dataset as csv files
fake_df_small.to_csv("./dataset/fake_small.csv", index=False)
true_df_small.to_csv("./dataset/true_small.csv", index=False)


# loading the trimmed dataset
trimmed_fake_df = pd.read_csv("./dataset/fake_small.csv")
trimmed_true_df = pd.read_csv("./dataset/true_small.csv")



# Add labels: 1 for fake, 0 for true
trimmed_fake_df['label'] = 1
trimmed_true_df['label'] = 0

# Combine the datasets
combined_df = pd.concat([trimmed_fake_df, trimmed_true_df], ignore_index=True)


# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(combined_df)


# Train/test split
dataset = dataset.train_test_split(test_size=0.2)

# Load pretrained multilingual tokenizer and model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    save_strategy="epoch",
    use_cpu=True,
    dataloader_num_workers=0,
    logging_steps=50,
    save_steps=500

)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics

)

# Train the model
trainer.train()

# Save fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuned model saved successfully")
