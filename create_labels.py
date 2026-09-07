import pandas as pd
import json

INPUT_FILE = "data/hindi_indicbert.csv"

df = pd.read_csv(INPUT_FILE)

# Get categories in alphabetical order
categories = sorted(df["label"].unique())

# Create mappings
label2id = {
    label: idx
    for idx, label in enumerate(categories)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

print("=" * 60)
print("LABEL MAPPING")
print("=" * 60)

for label, idx in label2id.items():
    print(f"{idx:2d} -> {label}")


print("\nNumber of labels:", len(categories))


# Save mappings
with open("data/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "label2id": label2id,
            "id2label": id2label
        },
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nSaved to:")
print("data/label_mapping.json")