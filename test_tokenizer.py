from transformers import AutoTokenizer

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

print("=" * 60)
print("LOADING INDICBERTv2 TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer loaded successfully!")

print("\nTokenizer class:")
print(type(tokenizer))

print("\nVocabulary size:")
print(tokenizer.vocab_size)


# Test Hindi sentence
text = "भारत सरकार ने नई योजना शुरू की।"

print("\nOriginal text:")
print(text)

# Tokenize
tokens = tokenizer.tokenize(text)

print("\nTokens:")
print(tokens)

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nToken IDs:")
print(token_ids)


# Full encoding
encoded = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=128
)

print("\nEncoded output:")
print(encoded)