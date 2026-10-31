import io
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)

st.set_page_config(page_title="Bipolar Molecular Pattern Analysis", page_icon="🧬", layout="wide")

st.title("🧬 Sequence-Based Prediction of Bipolar Disorder-Associated Molecular Pattern")
st.caption("Exploratory ML dashboard using RNA-seq-derived gene-level count data. This is a research prototype, not a clinical diagnostic tool.")

with st.expander("How to use this dashboard", expanded=True):
    st.markdown("""
    1. Upload the **GSE124326 count matrix** (`.txt`, `.tsv`, or `.csv`). It should contain one gene-ID column and one column per sample.
    2. Upload a **sample metadata CSV** with two columns: `Sample` and `Diagnosis`. `Sample` must match the sample titles in the count matrix (the app also removes a trailing `.counts`). `Diagnosis` must contain `Bipolar` or `Control`.
    3. Set filtering and feature-selection options, then click **Run analysis**.
    4. Use the prediction tab to upload a new matrix with the same gene IDs if you want to predict samples using the trained model.
    """)

def read_table(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    # GEO count matrices are commonly tab-separated text files
    return pd.read_csv(uploaded, sep="\t", comment="#", low_memory=False)

def normalize_sample_name(x):
    return str(x).strip().replace(".counts", "")

def canonical_label(x):
    s = str(x).strip().lower()
    if s in {"bipolar", "bp", "bp1", "bp2", "bipolar disorder"} or s.startswith("bp"):
        return "Bipolar"
    if s in {"control", "healthy control", "healthy"}:
        return "Control"
    return None

def build_pipeline(model_name, k, random_state):
    if model_name == "SVM":
        classifier = SVC(kernel="rbf", probability=True, random_state=random_state)
        steps = [("select", SelectKBest(score_func=f_classif, k=k)),
                 ("model", classifier)]
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1
        )
        steps = [("select", SelectKBest(score_func=f_classif, k=k)),
                 ("model", classifier)]
    else:
        classifier = LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        steps = [("select", SelectKBest(score_func=f_classif, k=k)),
                 ("model", classifier)]
    return Pipeline(steps)

def get_positive_prob(pipe, X):
    classes = list(pipe.classes_)
    idx = classes.index("Bipolar")
    return pipe.predict_proba(X)[:, idx]

count_file = st.file_uploader("1. Upload gene-level count matrix", type=["txt", "tsv", "csv"], key="counts")
meta_file = st.file_uploader("2. Upload sample metadata CSV (Sample, Diagnosis)", type=["csv"], key="metadata")

# The expo package includes the verified GSE124326 inputs, so the app can run immediately.
BASE_DIR = Path(__file__).resolve().parent
BUNDLED_COUNT = BASE_DIR / "GSE124326_count_matrix.txt"
BUNDLED_META = BASE_DIR / "sample_metadata.csv"
use_bundled = (not count_file and not meta_file and BUNDLED_COUNT.exists() and BUNDLED_META.exists())
if use_bundled:
    st.info("Using the included GSE124326 count matrix and verified sample metadata. Upload different files above to replace them.")
count_source = count_file if count_file else (BUNDLED_COUNT if use_bundled else None)
meta_source = meta_file if meta_file else (BUNDLED_META if use_bundled else None)

with st.sidebar:
    st.header("Analysis settings")
    min_count = st.number_input("Minimum count threshold", min_value=0, value=10, step=1)
    min_fraction = st.slider("Minimum fraction of samples meeting threshold", 0.05, 1.0, 0.10, 0.05)
    n_features = st.number_input("Number of selected features", min_value=5, max_value=5000, value=500, step=50)
    test_size = st.slider("Test set proportion", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    chosen_model = st.selectbox("Model for detailed view", ["Logistic Regression", "SVM", "Random Forest"])

if count_source is not None and meta_source is not None:
    try:
        raw = read_table(count_source)
        metadata = pd.read_csv(meta_source)
        if raw.shape[1] < 3:
            st.error("The count matrix must contain a gene-ID column and multiple sample columns.")
            st.stop()
        if metadata.shape[1] < 2:
            st.error("Metadata needs at least two columns: Sample and Diagnosis.")
            st.stop()

        # Use named columns when available; otherwise use the first two columns.
        sample_col = "Sample" if "Sample" in metadata.columns else metadata.columns[0]
        diagnosis_col = "Diagnosis" if "Diagnosis" in metadata.columns else metadata.columns[1]
        metadata = metadata[[sample_col, diagnosis_col]].copy()
        metadata.columns = ["Sample", "Diagnosis"]
        metadata["Sample_clean"] = metadata["Sample"].map(normalize_sample_name)
        metadata["Diagnosis"] = metadata["Diagnosis"].map(canonical_label)
        metadata = metadata.dropna(subset=["Diagnosis", "Sample_clean"]).drop_duplicates("Sample_clean")

        gene_col = "gene" if "gene" in raw.columns else raw.columns[0]
        raw[gene_col] = raw[gene_col].astype(str)
        raw = raw.drop_duplicates(subset=[gene_col]).set_index(gene_col)
        raw.columns = [normalize_sample_name(c) for c in raw.columns]
        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        raw = raw.clip(lower=0)

        # Keep only samples present in both the count matrix and metadata.
        shared = [c for c in raw.columns if c in set(metadata["Sample_clean"])]
        if not shared:
            st.error("No sample names matched. Check that metadata Sample values match the count-matrix column names.")
            st.info("Example count-matrix sample names: " + ", ".join(raw.columns[:5]))
            st.stop()
        label_map = metadata.set_index("Sample_clean")["Diagnosis"].to_dict()
        X_counts = raw[shared].T
        y = pd.Series([label_map[s] for s in shared], index=shared, name="Diagnosis")
        valid = y.isin(["Bipolar", "Control"])
        X_counts, y = X_counts.loc[valid], y.loc[valid]

        # Filter low-expression genes, then CPM-normalize and log transform.
        min_samples = max(1, int(np.ceil(min_fraction * X_counts.shape[0])))
        keep = (X_counts >= min_count).sum(axis=0) >= min_samples
        X_filtered = X_counts.loc[:, keep]
        library_size = X_filtered.sum(axis=1).replace(0, np.nan)
        X_cpm = X_filtered.div(library_size, axis=0) * 1_000_000
        X_log = np.log2(X_cpm.fillna(0) + 1)

        counts = y.value_counts()
        st.subheader("Dataset overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matched samples", f"{len(y):,}")
        c2.metric("Bipolar", f"{int(counts.get('Bipolar', 0)):,}")
        c3.metric("Control", f"{int(counts.get('Control', 0)):,}")
        c4.metric("Genes after filtering", f"{X_log.shape[1]:,}")

        if y.nunique() != 2 or counts.min() < 2:
            st.error("Both Bipolar and Control labels need at least two matched samples.")
            st.stop()

        st.caption(f"Initial genes: {raw.shape[0]:,} · Samples matched: {len(y):,} · Features after low-expression filtering: {X_log.shape[1]:,}")
        st.write("Class distribution")
        st.bar_chart(counts)

        if st.button("Run analysis", type="primary"):
            # Split samples before fitting feature selection to avoid test-set leakage.
            X_train, X_test, y_train, y_test = train_test_split(
                X_log, y, test_size=float(test_size), random_state=int(random_state), stratify=y
            )
            k = min(int(n_features), X_train.shape[1])
            model_names = ["SVM", "Random Forest", "Logistic Regression"]
            pipelines = {}
            rows = []
            predictions = {}
            probabilities = {}
            for name in model_names:
                pipe = build_pipeline(name, k, int(random_state))
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                prob = get_positive_prob(pipe, X_test)
                pipelines[name] = pipe
                predictions[name] = pred
                probabilities[name] = prob
                y_binary = (y_test == "Bipolar").astype(int)
                rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision (Bipolar)": precision_score(y_test, pred, pos_label="Bipolar", zero_division=0),
                    "Recall (Bipolar)": recall_score(y_test, pred, pos_label="Bipolar", zero_division=0),
                    "F1-score (Bipolar)": f1_score(y_test, pred, pos_label="Bipolar", zero_division=0),
                    "ROC-AUC": roc_auc_score(y_binary, prob)
                })

            results = pd.DataFrame(rows).set_index("Model")

            # Five-fold CV is performed only on the training partition.
            # The selector remains inside each pipeline, so it is refit within each CV fold.
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
            cv_rows = []
            for name in model_names:
                cv_pipe = build_pipeline(name, k, int(random_state))
                cv_scores = cross_validate(
                    cv_pipe, X_train, y_train, cv=cv,
                    scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
                    return_train_score=False, n_jobs=1
                )
                cv_rows.append({
                    "Model": name,
                    "CV Accuracy (mean)": cv_scores["test_accuracy"].mean(),
                    "CV Accuracy (SD)": cv_scores["test_accuracy"].std(),
                    "CV ROC-AUC (mean)": cv_scores["test_roc_auc"].mean(),
                    "CV ROC-AUC (SD)": cv_scores["test_roc_auc"].std()
                })
            cv_results = pd.DataFrame(cv_rows).set_index("Model")
            st.session_state["analysis"] = {
                "X_log": X_log, "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "pipelines": pipelines, "results": results, "cv_results": cv_results,
                "predictions": predictions, "probabilities": probabilities,
                "gene_names": list(X_log.columns), "k": k, "sample_names": list(shared)
            }

        if "analysis" in st.session_state:
            a = st.session_state["analysis"]
            st.subheader("Model comparison — held-out test set")
            st.dataframe(a["results"].style.format("{:.4f}"), use_container_width=True)
            st.subheader("Five-fold cross-validation — training data only")
            st.dataframe(a["cv_results"].style.format("{:.4f}"), use_container_width=True)
            best_model = a["cv_results"]["CV ROC-AUC (mean)"].idxmax()
            st.success(f"Highest mean CV ROC-AUC: {best_model} ({a['cv_results'].loc[best_model, 'CV ROC-AUC (mean)']:.4f}). Use the held-out test set for final reporting; treat this as model-selection evidence, not a clinical validation.")

            left, right = st.columns(2)
            with left:
                st.markdown(f"**{chosen_model}: confusion matrix**")
                pred = a["predictions"][chosen_model]
                cm = confusion_matrix(a["y_test"], pred, labels=["Bipolar", "Control"])
                st.dataframe(pd.DataFrame(cm, index=["Actual Bipolar", "Actual Control"],
                                          columns=["Predicted Bipolar", "Predicted Control"]))
            with right:
                st.markdown(f"**{chosen_model}: classification report**")
                report = classification_report(a["y_test"], a["predictions"][chosen_model], output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

            st.subheader("ROC curves")
            fig, ax = plt.subplots()
            y_binary = (a["y_test"] == "Bipolar").astype(int)
            for name in ["SVM", "Random Forest", "Logistic Regression"]:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_binary, a["probabilities"][name])
                ax.plot(fpr, tpr, label=f"{name} (AUC={a['results'].loc[name, 'ROC-AUC']:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # Show selected features and a simple model-specific importance ranking.
            st.subheader("Selected molecular features")
            pipe = a["pipelines"][chosen_model]
            selector = pipe.named_steps["select"]
            selected_names = np.array(a["gene_names"])[selector.get_support()]
            scores = selector.scores_[selector.get_support()]
            feature_table = pd.DataFrame({"Gene/feature": selected_names, "ANOVA F-score": scores})
            feature_table = feature_table.sort_values("ANOVA F-score", ascending=False).head(30)
            st.dataframe(feature_table, use_container_width=True)
            st.caption("These are statistically selected features, not validated biomarkers. Biological interpretation requires additional analysis.")

            # Download predictions/results for the current held-out test set.
            pred_df = pd.DataFrame({
                "Sample": a["y_test"].index,
                "Actual_Diagnosis": a["y_test"].values,
                "Predicted_Diagnosis": a["predictions"][chosen_model],
                "Bipolar_probability": a["probabilities"][chosen_model]
            })
            st.download_button("Download test predictions CSV", pred_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{chosen_model.lower().replace(' ', '_')}_test_predictions.csv",
                               mime="text/csv")
            st.download_button("Download model comparison CSV", a["results"].to_csv().encode("utf-8"),
                               file_name="model_comparison.csv", mime="text/csv")
            st.download_button("Download cross-validation CSV", a["cv_results"].to_csv().encode("utf-8"),
                               file_name="cross_validation_results.csv", mime="text/csv")

            st.subheader("Predict from a new count matrix (optional)")
            st.write("Upload a new count matrix with the same gene IDs and sample columns. The app aligns genes to the training features and applies the trained preprocessing/selection/model pipeline.")
            new_file = st.file_uploader("New count matrix", type=["txt", "tsv", "csv"], key="new_counts")
            if new_file:
                try:
                    new_raw = read_table(new_file)
                    new_gene_col = "gene" if "gene" in new_raw.columns else new_raw.columns[0]
                    new_raw[new_gene_col] = new_raw[new_gene_col].astype(str)
                    new_raw = new_raw.drop_duplicates(subset=[new_gene_col]).set_index(new_gene_col)
                    new_raw = new_raw.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
                    common_genes = [g for g in a["X_log"].columns if g in new_raw.index]
                    if len(common_genes) == 0:
                        st.error("No overlapping gene IDs found between the new matrix and the training data.")
                    else:
                        new_samples = new_raw[common_genes].T
                        # Reproduce the CPM/log transform on the new matrix.
                        lib = new_samples.sum(axis=1).replace(0, np.nan)
                        new_cpm = new_samples.div(lib, axis=0) * 1_000_000
                        new_log = np.log2(new_cpm.fillna(0) + 1)
                        # Add any training genes absent from the new matrix as zero columns.
                        new_log = new_log.reindex(columns=a["X_log"].columns, fill_value=0)
                        model = a["pipelines"][chosen_model]
                        new_pred = model.predict(new_log)
                        new_prob = get_positive_prob(model, new_log)
                        out_df = pd.DataFrame({"Sample": new_log.index, "Predicted_Diagnosis": new_pred,
                                               "Bipolar_probability": new_prob})
                        st.dataframe(out_df, use_container_width=True)
                        st.download_button("Download new-sample predictions", out_df.to_csv(index=False).encode("utf-8"),
                                           file_name="new_sample_predictions.csv", mime="text/csv")
                except Exception as exc:
                    st.error(f"Could not process the new matrix: {exc}")

    except Exception as exc:
        st.error(f"Could not read/process the uploaded files: {exc}")
        st.info("Check the file delimiters, gene-ID column, and metadata labels. The metadata needs one row per sample.")
else:
    st.info("Upload both the count matrix and metadata CSV to start. The expo package includes these verified files and will load them automatically when available.")
    st.markdown("""
    **Metadata format example**

    | Sample | Diagnosis |
    |---|---|
    | 089357B | Control |
    | 089366A | Bipolar |
    | 089412B | Bipolar |

    Diagnosis labels such as `BP1` and `BP2` are combined as `Bipolar`.
    """)

st.divider()
st.caption("Research prototype only. Predictions are not medical advice and must not be used to diagnose or make treatment decisions.")
