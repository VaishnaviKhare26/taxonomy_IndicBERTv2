from sklearn.metrics import classification_report

test_df = pd.read_csv("data/hindi_test.csv")
test_df["label"] = test_df["label"].map(label2id)
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
test_dataset = test_dataset.map(tokenize_batch, batched=True)

print("Evaluating on Test Set...")
predictions_output = trainer.predict(test_dataset)
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = predictions_output.label_ids

print("\n" + "=" * 60)
print("TEST SET CLASSIFICATION REPORT")
print("=" * 60)
print(
    classification_report(
        y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)]
    )
)