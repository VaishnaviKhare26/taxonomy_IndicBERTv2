import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

FILE = "data/news.hindi_title_content_category.csv"

df = pd.read_csv(FILE)

print("=" * 60)
print("HINDI DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCategory distribution:")
print(df["ai_analysis.category"].value_counts())

print("\nNumber of categories:")
print(df["ai_analysis.category"].nunique())