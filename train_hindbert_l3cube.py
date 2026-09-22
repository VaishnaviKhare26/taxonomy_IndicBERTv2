"""
=============================================================================
TAXONOMY CLASSIFICATION BENCHMARK: L3Cube HindBERT (l3cube-pune/hindi-bert-v2)
=============================================================================
L3Cube-Pune's Monolingual Hindi BERT model (~110M parameters)
Trained exclusively on large Devanagari Hindi corpora.

Requirements (run in Google Colab before executing):
!pip install -q transformers datasets evaluate scikit-learn seaborn matplotlib accelerate sentencepiece protobuf
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 1. Configuration & Hyperparameters (Matches IndicBERTv2 Baseline)
# ---------------------------------------------------------------------------
# You can also test "l3cube-pune/hindi-bert-scratch" or "l3cube-pune/dev-bert"
MODEL_NAME = "l3cube-pune/hindi-bert-v2"
MODEL_DISPLAY_NAME = "L3Cube HindBERT (l3cube-pune/hindi-bert-v2)"
OUTPUT_DIR = "./results_hindbert"
BEST_MODEL_DIR = "./best_hindbert_hindi_model"
CM_OUTPUT_FILE = "hindbert_confusion_matrix.png"
REPORT_OUTPUT_FILE = "hindbert_classification_report.txt"
METRICS_OUTPUT_FILE = "hindbert_metrics.json"

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
SEED = 42


# ---------------------------------------------------------------------------
# 2. Data Resolution and Preparation
# ---------------------------------------------------------------------------
def resolve_and_load_data():
    """
    Intelligently locates the dataset:
    1. Looks for existing train/val/test splits and label mapping.
    2. If not found, looks for raw news.hindi csv, pre-processes, and splits
       with the identical 70/15/15 ratio and random_state=42.
    """
    possible_dirs = [Path("data"), Path(".")]
    train_file = None
    val_file = None
    test_file = None
    mapping_file = None

    for d in possible_dirs:
        tf = d / "hindi_train.csv"
        vf = d / "hindi_validation.csv"
        tef = d / "hindi_test.csv"
        mf = d / "label_mapping.json"
        if tf.exists() and vf.exists() and tef.exists() and mf.exists():
            train_file, val_file, test_file, mapping_file = tf, vf, tef, mf
            break

    if train_file is not None:
        print(f"Found existing pre-split datasets in: {train_file.parent}")
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        label2id = mapping_data["label2id"]
        id2label = {int(k): v for k, v in mapping_data["id2label"].items()}
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        return train_df, val_df, test_df, label2id, id2label

    raw_candidates = [
        Path("data/news.hindi_title_content_category.csv"),
        Path("news.hindi_title_content_category.csv"),
        Path("data/news.hindi_title_content.csv"),
        Path("news.hindi_title_content.csv"),
        Path("data/hindi_indicbert.csv"),
        Path("hindi_indicbert.csv")
    ]
    raw_file = None
    for cand in raw_candidates:
        if cand.exists():
            raw_file = cand
            break

    if raw_file is None:
        raise FileNotFoundError(
            "Could not locate dataset files! Please upload 'hindi_train.csv', "
            "'hindi_validation.csv', 'hindi_test.csv', 'label_mapping.json' "
            "OR 'news.hindi_title_content_category.csv' / 'news.hindi_title_content.csv'."
        )

    print(f"Pre-split files not found. Creating splits from raw dataset: {raw_file}")
    df = pd.read_csv(raw_file)

    cat_col = None
    for col in ["ai_analysis.category", "label", "category"]:
        if col in df.columns:
            cat_col = col
            break
    if cat_col is None:
        raise ValueError(f"Could not identify category column among {df.columns.tolist()}")

    if "text" not in df.columns:
        df["title"] = df["title"].fillna("") if "title" in df.columns else ""
        df["content"] = df["content"].fillna("") if "content" in df.columns else ""
        df["text"] = df["title"].astype(str).str.strip() + "\n\n" + df["content"].astype(str).str.strip()

    df = df.dropna(subset=[cat_col]).copy()
    df["label"] = df[cat_col].astype(str).str.strip()
    df = df[df["text"].str.strip().str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text"]).copy()

    cat_counts = df["label"].value_counts()
    valid_categories = cat_counts[cat_counts >= 20].index
    df = df[df["label"].isin(valid_categories)].copy()
    df = df[["text", "label"]].copy()

    categories = sorted(df["label"].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(categories)}
    id2label = {idx: lbl for idx, lbl in enumerate(categories)}

    # Stratified 70/15/15 split
    train_val, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
    train_df, val_df = train_test_split(train_val, test_size=0.17647, random_state=SEED, stratify=train_val["label"])

    save_dir = Path("data") if Path("data").exists() else Path(".")
    train_df.to_csv(save_dir / "hindi_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(save_dir / "hindi_validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(save_dir / "hindi_test.csv", index=False, encoding="utf-8-sig")
    with open(save_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, ensure_ascii=False, indent=4)

    return train_df, val_df, test_df, label2id, id2label


def main():
    print("=" * 70)
    print(f"STARTING TAXONOMY BENCHMARK: {MODEL_DISPLAY_NAME}")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df, label2id, id2label = resolve_and_load_data()
    num_labels = len(label2id)

    print(f"Classes ({num_labels}): {list(label2id.keys())}")
    print(f"Split sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # Map string labels to integer IDs
    train_df["label"] = train_df["label"].map(label2id)
    val_df["label"] = val_df["label"].map(label2id)
    test_df["label"] = test_df["label"].map(label2id)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    # 3. Load Tokenizer & Tokenize
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Load Model
    print(f"\nLoading model: {MODEL_NAME} with {num_labels} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 5. Metrics & Training Setup
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro")
        weighted_f1 = f1_score(labels, predictions, average="weighted")
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }

    try:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=25,
            save_total_limit=1,
            seed=SEED,
            report_to="none"
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=25,
            save_total_limit=1,
            seed=SEED,
            report_to="none"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 6. Train Model
    print("\n" + "=" * 70)
    print(f"TRAINING {MODEL_DISPLAY_NAME}")
    print("=" * 70)
    trainer.train()

    # Save best model
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    print(f"\nBest model saved to {BEST_MODEL_DIR}")

    # 7. Evaluate on Held-Out Test Set
    print("\n" + "=" * 70)
    print(f"EVALUATING {MODEL_DISPLAY_NAME} ON HELD-OUT TEST SET")
    print("=" * 70)

    predictions_output = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids

    class_names = [id2label[i] for i in range(num_labels)]

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro")
    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    clf_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n" + "=" * 70)
    print(f"TEST SET CLASSIFICATION REPORT: {MODEL_DISPLAY_NAME}")
    print("=" * 70)
    print(f"Accuracy:    {test_acc:.4f}")
    print(f"Macro F1:    {test_macro_f1:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}\n")
    print(clf_report)

    # Save text report
    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"MODEL: {MODEL_DISPLAY_NAME}\n")
        f.write(f"Accuracy:    {test_acc:.4f}\n")
        f.write(f"Macro F1:    {test_macro_f1:.4f}\n")
        f.write(f"Weighted F1: {test_weighted_f1:.4f}\n\n")
        f.write(clf_report)
    print(f"Report saved to: {REPORT_OUTPUT_FILE}")

    # Save JSON metrics
    metrics_summary = {
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1)
    }
    with open(METRICS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics JSON saved to: {METRICS_OUTPUT_FILE}")

    # 8. Generate & Save Normalized Confusion Matrix Plot
    print("\nGenerating Normalized Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.title(f"{MODEL_DISPLAY_NAME} - Normalized Confusion Matrix", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(CM_OUTPUT_FILE, dpi=300)
    plt.show()

    print(f"\n[SUCCESS] Confusion matrix saved to: {CM_OUTPUT_FILE}")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
