import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------
# Configuration & Hyperparameters
# ---------------------------------------------------------
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
LABEL_MAPPING_FILE = "data/label_mapping.json"
TRAIN_FILE = "data/hindi_train.csv"
VAL_FILE = "data/hindi_validation.csv"
OUTPUT_DIR = "./results_indicbert_hindi"
MAX_LENGTH = 512

print("Device using:", torch.cuda.get_device_name(0))

# 1. Load Label Mappings
with open(LABEL_MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

label2id = mapping_data["label2id"]
id2label = {int(k): v for k, v in mapping_data["id2label"].items()}
num_labels = len(label2id)

# 2. Load Datasets
train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

train_df["label"] = train_df["label"].map(label2id)
val_df["label"] = val_df["label"].map(label2id)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

# 3. Load Tokenizer & Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Load Model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
)


# 5. Evaluation Metrics Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# 6. Training Arguments (Optimized for T4 GPU)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=True,  # Fast half-precision on GPU
    logging_steps=25,
    save_total_limit=1,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting training on T4 GPU...")
trainer.train()

# 7. Save the best model
final_model_path = "./best_indicbert_hindi_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nTraining complete. Model saved to {final_model_path}")