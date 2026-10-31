# Bipolar Molecular Pattern ML Dashboard — Expo Package

## What's included
- `app.py` — Streamlit ML dashboard
- `GSE124326_count_matrix.txt` — supplied gene-level RNA-seq count matrix
- `sample_metadata.csv` — verified diagnosis metadata (480 samples)
- `requirements.txt` — Python dependencies
- `run_dashboard.command` — Mac launcher
- `heldout_test_model_comparison.csv` — metrics recalculated from the supplied files
- `five_fold_cv_training_results.csv` — 5-fold CV results on the training partition
- `heldout_test_predictions.csv` — held-out test predictions for all three models

## Run on macOS
1. Extract this ZIP.
2. Open Terminal and `cd` into the extracted folder.
3. Run `python3 -m pip install -r requirements.txt`.
4. Run `python3 -m streamlit run app.py`.
5. Open the local URL printed by Streamlit (usually http://localhost:8501).

You can also double-click `run_dashboard.command`; macOS may require allowing it in Privacy & Security, or run it from Terminal.

The dashboard automatically loads the included dataset and metadata when no replacement uploads are selected. Click **Run analysis**.

## Verified input details
- Count matrix: 57,820 genes × 480 samples (plus gene-ID column)
- Metadata: 480 rows; 240 Bipolar and 240 Control
- Matched sample IDs: 480/480
- Low-expression filter: count >= 10 in at least 10% of samples
- Retained genes after filtering: 17,802
- Normalization: CPM followed by log2(CPM + 1)
- Hold-out split: stratified 80:20 (384 train, 96 test)
- Feature selection: SelectKBest with ANOVA F-test, k=500; fit within training folds

## Scientific / clinical limitations
The input is a processed gene-level RNA-seq count matrix, not raw FASTQ reads or direct k-mer sequence features. This is an exploratory molecular/gene-expression classification prototype, not a clinically validated diagnostic system. Selected statistical features are not validated biomarkers. Do not use predictions for medical decisions.

## Reproducibility
The application follows the Colab model settings: SVM with RBF kernel and probability estimates; Random Forest with 200 trees and balanced class weights; Logistic Regression with max_iter=1000. Random seed is 42 by default. The dashboard recalculates metrics and also runs 5-fold cross-validation on the training partition.
