import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

TRAIN_FILE = "data/hindi_train.csv"
VAL_FILE = "data/hindi_validation.csv"
TEST_FILE = "data/hindi_test.csv"

print("=" * 60)
print("LOADING INDICBERTv2 TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer loaded successfully!")
print("Vocabulary size:", tokenizer.vocab_size)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ---------------------------------------------------------
# Function to calculate token lengths
# ---------------------------------------------------------

def get_token_lengths(df):

    lengths = []

    for text in df["text"]:

        tokens = tokenizer(
            str(text),
            truncation=False,
            add_special_tokens=True
        )

        lengths.append(len(tokens["input_ids"]))

    return lengths


# ---------------------------------------------------------
# Calculate token lengths
# ---------------------------------------------------------

print("\nCalculating token lengths...")

train_lengths = get_token_lengths(train_df)
val_lengths = get_token_lengths(val_df)
test_lengths = get_token_lengths(test_df)


# ---------------------------------------------------------
# Display statistics
# ---------------------------------------------------------

def show_statistics(name, lengths):

    series = pd.Series(lengths)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Minimum tokens:", series.min())
    print("Maximum tokens:", series.max())
    print("Average tokens:", round(series.mean(), 2))
    print("Median tokens:", series.median())

    print("90th percentile:", int(series.quantile(0.90)))
    print("95th percentile:", int(series.quantile(0.95)))
    print("99th percentile:", int(series.quantile(0.99)))

    print("\nArticles <= 128 tokens:", (series <= 128).sum())
    print("Articles <= 256 tokens:", (series <= 256).sum())
    print("Articles <= 512 tokens:", (series <= 512).sum())


show_statistics("TRAIN TOKEN LENGTHS", train_lengths)
show_statistics("VALIDATION TOKEN LENGTHS", val_lengths)
show_statistics("TEST TOKEN LENGTHS", test_lengths)


print("\n" + "=" * 60)
print("TOKEN LENGTH ANALYSIS COMPLETE")
print("=" * 60)