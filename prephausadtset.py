import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('./dataset/hausa.csv')

# Remove rows with -1 labels
df = df[df['label'] != -1]

# Rename columns for Hugging Face compatibility
df = df.rename(columns={'NewsText': 'text'})

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Create datasets
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

# Convert to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Save prepared datasets locally
dataset_dict.save_to_disk('./dataset/hausa_fake_news_dataset')
