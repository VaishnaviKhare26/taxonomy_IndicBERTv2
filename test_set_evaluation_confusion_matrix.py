import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix

# Load and tokenize test data
test_df = pd.read_csv("data/hindi_test.csv")
test_df["label"] = test_df["label"].map(label2id)
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
test_dataset = test_dataset.map(tokenize_batch, batched=True)

# Generate predictions
print("Running inference on test set...")
predictions_output = trainer.predict(test_dataset)
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = predictions_output.label_ids

class_names = [id2label[i] for i in range(num_labels)]

# 1. Classification Report
print("\n" + "=" * 60)
print("TEST SET CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 2. Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred, normalize="true")

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title(
    "IndicBERTv2 Hindi News Classification - Normalized Confusion Matrix",
    fontsize=14,
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("hindi_confusion_matrix.png", dpi=300)
plt.show()
print("Confusion matrix saved to hindi_confusion_matrix.png")