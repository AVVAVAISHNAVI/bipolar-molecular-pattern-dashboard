# Bipolar Molecular Pattern ML Dashboard

A Streamlit research dashboard for exploratory classification of Bipolar vs Control samples using processed RNA-seq-derived gene-level count data from NCBI GEO GSE124326.

## Included files
- `app.py` — Streamlit dashboard application
- `GSE124326_preprocessed.npz` — compact, preprocessed 480-sample by 17,802-feature matrix with labels and gene IDs
- `sample_metadata.csv` — matched sample-level diagnosis metadata
- `GSE124326_count_matrix.txt` — original public gene-level count matrix (retained for reproducibility)
- `requirements.txt` — Python dependencies

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud deployment
Repository: `AVVAVAISHNAVI/bipolar-molecular-pattern-dashboard`
Branch: `main`
Main file: `app.py`

The dashboard loads `GSE124326_preprocessed.npz` by default, so users do not need to upload the large count matrix during the demo. The app fits three pipelines, reports a held-out test comparison, performs five-fold cross-validation on the training partition, shows ROC/confusion matrix/selected statistical features, and supports prediction on a compatible new count matrix.

## Data processing represented in the preprocessed file
- 480 samples: 240 Bipolar and 240 Control
- 57,820 original gene rows
- Filter genes with count >= 10 in at least 10% of samples
- 17,802 genes retained
- Per-sample CPM normalization
- log2(CPM + 1) transformation
- 500 ANOVA F-test-selected features are selected within training folds by the dashboard pipeline

## Scientific and ethical limitations
The uploaded file is a processed gene-level RNA-seq count matrix, not raw FASTQ reads. The dashboard is an exploratory research prototype, not a medical diagnostic system. Statistical feature ranking does not by itself establish a validated biomarker or biological cause. Results require independent cohort validation and biological interpretation.
