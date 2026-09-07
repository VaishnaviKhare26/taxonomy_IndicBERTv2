import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/hindi_indicbert.csv"

TRAIN_FILE = "data/hindi_train.csv"
VAL_FILE = "data/hindi_validation.csv"
TEST_FILE = "data/hindi_test.csv"

RANDOM_STATE = 42

print("=" * 60)
print("LOADING PREPARED HINDI DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nCategory distribution:")
print(df["label"].value_counts())


# ---------------------------------------------------------
# STEP 1: Split into 85% temporary data and 15% test data
# ---------------------------------------------------------

train_val, test = train_test_split(
    df,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# ---------------------------------------------------------
# STEP 2: Split the remaining 85% into
#         70% train and 15% validation
# ---------------------------------------------------------

# Validation should be 15% of the ORIGINAL dataset.
# Therefore, 15 / 85 = approximately 17.65%.

train, validation = train_test_split(
    train_val,
    test_size=0.17647,
    random_state=RANDOM_STATE,
    stratify=train_val["label"]
)


# ---------------------------------------------------------
# STEP 3: Save datasets
# ---------------------------------------------------------

train.to_csv(
    TRAIN_FILE,
    index=False,
    encoding="utf-8-sig"
)

validation.to_csv(
    VAL_FILE,
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    TEST_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------
# STEP 4: Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT RESULTS")
print("=" * 60)

print("\nTrain:")
print(f"Articles: {len(train)}")
print(f"Percentage: {len(train) / len(df) * 100:.2f}%")

print("\nValidation:")
print(f"Articles: {len(validation)}")
print(f"Percentage: {len(validation) / len(df) * 100:.2f}%")

print("\nTest:")
print(f"Articles: {len(test)}")
print(f"Percentage: {len(test) / len(df) * 100:.2f}%")


print("\n" + "=" * 60)
print("TRAIN CATEGORY DISTRIBUTION")
print("=" * 60)

print(train["label"].value_counts())


print("\n" + "=" * 60)
print("VALIDATION CATEGORY DISTRIBUTION")
print("=" * 60)

print(validation["label"].value_counts())


print("\n" + "=" * 60)
print("TEST CATEGORY DISTRIBUTION")
print("=" * 60)

print(test["label"].value_counts())


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)