import pandas as pd
from pathlib import Path
print("hello")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/news.hindi_title_content_category.csv"
OUTPUT_FILE = "data/hindi_indicbert.csv"

MIN_SAMPLES = 20


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING HINDI DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:")
print(df.shape)

# ============================================================
# 2. HANDLE MISSING TEXT
# ============================================================

df["title"] = df["title"].fillna("")
df["content"] = df["content"].fillna("")


# ============================================================
# 3. REMOVE MISSING LABELS
# ============================================================

df = df.dropna(
    subset=["ai_analysis.category"]
).copy()


# ============================================================
# 4. CREATE MODEL INPUT
# ============================================================

df["text"] = (
    df["title"].astype(str).str.strip()
    + "\n\n"
    + df["content"].astype(str).str.strip()
)


# ============================================================
# 5. REMOVE COMPLETELY EMPTY TEXT
# ============================================================

df = df[
    df["text"].str.strip().str.len() > 0
].copy()


# ============================================================
# 6. REMOVE DUPLICATE ARTICLES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["text"]
).copy()

duplicates_removed = before - len(df)

print("\nDuplicate articles removed:")
print(duplicates_removed)


# ============================================================
# 7. CLEAN LABELS
# ============================================================

df["label"] = (
    df["ai_analysis.category"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. SHOW ORIGINAL CATEGORY DISTRIBUTION
# ============================================================

print("\nOriginal category distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# 9. REMOVE VERY RARE CATEGORIES
# ============================================================

category_counts = df["label"].value_counts()

valid_categories = category_counts[
    category_counts >= MIN_SAMPLES
].index


df = df[
    df["label"].isin(valid_categories)
].copy()


# ============================================================
# 10. FINAL COLUMNS
# ============================================================

df = df[
    ["text", "label"]
].copy()


# ============================================================
# 11. SAVE
# ============================================================

Path("data").mkdir(
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 60)
print("FINAL INDICBERT DATASET")
print("=" * 60)

print("\nTotal articles:")
print(len(df))

print("\nNumber of categories:")
print(df["label"].nunique())

print("\nCategories retained:")

print(
    df["label"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)