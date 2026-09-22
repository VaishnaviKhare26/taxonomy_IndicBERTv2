# Google Colab Benchmarking Guide: Indic Taxonomy Classification

This guide explains how to run comparative experiments on Google Colab across 4 transformer architectures for Hindi news taxonomy classification:

1. **IndicBERTv2 Baseline**: `ai4bharat/IndicBERTv2-MLM-only` (~278M params) — *Already completed in `output_colab/`*
2. **MuRIL**: `google/muril-base-cased` (~237M params) — via `train_muril.py`
3. **IndicBERT v1**: `ai4bharat/indic-bert` (~33M params) — via `train_indicbert_v1.py`
4. **L3Cube HindBERT**: `l3cube-pune/hindi-bert-v2` (~110M params) — via `train_hindbert_l3cube.py`

---

## 1. Setup in Google Colab

### Step 1.1: Enable GPU Runtime
In Google Colab, go to:
**Runtime > Change runtime type > Hardware accelerator > T4 GPU** (or any available GPU).

### Step 1.2: Install Required Libraries
Run this in the first Colab cell:
```python
!pip install -q transformers datasets evaluate scikit-learn seaborn matplotlib accelerate sentencepiece protobuf
```

---

## 2. Uploading Data & Scripts

You have two simple options:

### Option A: Upload Files Directly via Colab Files Pane
- Upload either:
  - The `data/` folder containing `hindi_train.csv`, `hindi_validation.csv`, `hindi_test.csv`, and `label_mapping.json` (Recommended for exact 1:1 split parity).
  - OR just the raw CSV: `news.hindi_title_content_category.csv` / `news.hindi_title_content.csv`.
- Upload the script you want to run:
  - `train_muril.py`
  - `train_indicbert_v1.py`
  - `train_hindbert_l3cube.py`

### Option B: Clone your GitHub Repo Directly in Colab
```bash
!git clone https://github.com/VaishnaviKhare26/taxonomy_IndicBERTv2.git
%cd taxonomy_IndicBERTv2
```

---

## 3. Running the Benchmark Scripts

Each script is completely self-contained. It loads data, tokenizes, fine-tunes for 4 epochs, evaluates on the held-out 15% test set, prints the classification report, and displays & saves the confusion matrix heatmap.

### Run MuRIL (Google)
```bash
!python train_muril.py
```
- **Generated Outputs**:
  - `muril_confusion_matrix.png` (High-res 300 DPI normalized confusion matrix)
  - `muril_classification_report.txt`
  - `muril_metrics.json`
  - `./best_muril_hindi_model/`

### Run IndicBERT v1 (AI4Bharat)
```bash
!python train_indicbert_v1.py
```
- **Generated Outputs**:
  - `indicbert_v1_confusion_matrix.png` (High-res 300 DPI normalized confusion matrix)
  - `indicbert_v1_classification_report.txt`
  - `indicbert_v1_metrics.json`
  - `./best_indicbert_v1_hindi_model/`

### Run L3Cube HindBERT (L3Cube-Pune)
```bash
!python train_hindbert_l3cube.py
```
- **Generated Outputs**:
  - `hindbert_confusion_matrix.png` (High-res 300 DPI normalized confusion matrix)
  - `hindbert_classification_report.txt`
  - `hindbert_metrics.json`
  - `./best_hindbert_hindi_model/`

---

## 4. Benchmark Comparison Summary Template

Once you run all models, you can populate this benchmark table for your paper/report:

| Model Architecture | Hugging Face ID | Param Count | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IndicBERT v2 (Baseline)** | `ai4bharat/IndicBERTv2-MLM-only` | ~278M | **57.68%** | **34.56%** | *(In output_colab)* | *(In output_colab)* | *(In output_colab)* |
| **MuRIL** | `google/muril-base-cased` | ~237M | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **IndicBERT v1** | `ai4bharat/indic-bert` | ~33M | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **L3Cube HindBERT** | `l3cube-pune/hindi-bert-v2` | ~110M | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 5. Notes on Model Architectures & Behavior

1. **MuRIL (`google/muril-base-cased`)**:
   - Uses standard BERT-base architecture.
   - Trained on monolingual and translated/transliterated parallel text for 16 Indian languages.
   - Often exhibits strong cross-lingual generalization and Devanagari representation.

2. **IndicBERT v1 (`ai4bharat/indic-bert`)**:
   - ALBERT-style architecture (parameter-shared layers with 128 embedding projection).
   - Extremely lightweight (~33M parameters), train epochs complete significantly faster.
   - Requires SentencePiece tokenizer (`keep_accents=True` is handled automatically in the script).

3. **L3Cube HindBERT (`l3cube-pune/hindi-bert-v2`)**:
   - Monolingual BERT tailored specifically for Hindi.
   - Has a Hindi-specific vocabulary and pre-training distribution without competition from other languages' token budgets.
