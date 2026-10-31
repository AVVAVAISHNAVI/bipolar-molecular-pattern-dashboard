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
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# ---------------------------------------------------------------------
# Page setup and project description
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Bipolar Molecular Pattern Analysis",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Sequence-Based Prediction of Bipolar Disorder-Associated Molecular Pattern")
st.caption(
    "Exploratory ML dashboard using processed RNA-seq-derived gene-level count data. "
    "Research prototype only — not a clinical diagnostic tool."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "GSE124326_preprocessed.npz"

@st.cache_data(show_spinner="Loading preprocessed GSE124326 data...")
def load_preprocessed_data(path_string):
    data = np.load(path_string, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    y = data["y"].astype(str)
    samples = data["sample_names"].astype(str)
    genes = data["gene_names"].astype(str)
    return X, y, samples, genes

def build_pipeline(model_name, k, random_state):
    # Match the model configurations used in the Colab implementation.
    steps = [("select", SelectKBest(score_func=f_classif, k=k))]
    if model_name == "SVM":
        steps.append(("model", SVC(
            kernel="rbf", probability=True, random_state=random_state
        )))
    elif model_name == "Random Forest":
        steps.append(("model", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=random_state
        )))
    else:
        steps.append(("model", LogisticRegression(
            max_iter=1000, random_state=random_state
        )))
    return Pipeline(steps)

def positive_probability(model, X):
    classes = list(model.classes_)
    bipolar_index = classes.index("Bipolar")
    return model.predict_proba(X)[:, bipolar_index]

def bipolar_roc_auc_scorer(estimator, X_fold, y_fold):
    # Explicitly treat Bipolar as the positive class in every CV fold.
    probabilities = positive_probability(estimator, X_fold)
    y_binary = (np.asarray(y_fold) == "Bipolar").astype(int)
    return roc_auc_score(y_binary, probabilities)

def metric_row(y_true, y_pred, y_prob):
    y_binary = (y_true == "Bipolar").astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Bipolar)": precision_score(
            y_true, y_pred, pos_label="Bipolar", zero_division=0
        ),
        "Recall (Bipolar)": recall_score(
            y_true, y_pred, pos_label="Bipolar", zero_division=0
        ),
        "F1-score (Bipolar)": f1_score(
            y_true, y_pred, pos_label="Bipolar", zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_binary, y_prob)
    }

if not DATA_FILE.exists():
    st.error(
        "The bundled preprocessed dataset is missing. Ensure "
        "`GSE124326_preprocessed.npz` is committed in the same GitHub repository as app.py."
    )
    st.stop()

X, y, sample_names, gene_names = load_preprocessed_data(str(DATA_FILE))

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Analysis settings")
    test_fraction = st.slider("Test set proportion", 0.10, 0.40, 0.20, 0.05)
    feature_count = st.number_input(
        "Selected features", min_value=50, max_value=min(5000, X.shape[1]),
        value=min(500, X.shape[1]), step=50
    )
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    detailed_model = st.selectbox(
        "Detailed model view",
        ["Logistic Regression", "SVM", "Random Forest"]
    )
    run_button = st.button("Run / refresh analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------
label_counts = pd.Series(y).value_counts()
st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(y):,}")
c2.metric("Bipolar", f"{int(label_counts.get('Bipolar', 0)):,}")
c3.metric("Control", f"{int(label_counts.get('Control', 0)):,}")
c4.metric("Gene features", f"{X.shape[1]:,}")

with st.expander("Dataset and preprocessing details"):
    st.markdown("""
    - **Source:** NCBI GEO, accession **GSE124326**
    - **Input used here:** processed RNA-seq-derived gene-level count matrix
    - **Original matrix:** 57,820 gene rows × 480 samples
    - **Low-expression filtering:** 17,802 features retained
    - **Normalization:** Counts Per Million (CPM), followed by `log₂(CPM + 1)`
    - **Metadata:** 240 Bipolar and 240 Control samples; all 480 sample IDs matched
    - **Selected features:** ANOVA F-test (`SelectKBest`), fitted using training data only
    """)

left, right = st.columns([1, 1])
with left:
    st.markdown("**Class distribution**")
    st.bar_chart(label_counts)
with right:
    st.markdown("**Analysis pipeline**")
    st.markdown(
        "Data → filtering → CPM/log₂ normalization → stratified split → "
        "ANOVA feature selection → SVM / Random Forest / Logistic Regression → evaluation"
    )

# ---------------------------------------------------------------------
# Fit models when the user clicks the button; retain results across reruns
# ---------------------------------------------------------------------
if run_button or "dashboard_results" not in st.session_state:
    if y.min() == y.max() or len(np.unique(y)) != 2:
        st.error("The dataset must contain both Bipolar and Control classes.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_fraction),
        random_state=int(random_state),
        stratify=y
    )
    k = min(int(feature_count), X_train.shape[1])
    model_names = ["SVM", "Random Forest", "Logistic Regression"]

    fitted_models = {}
    predictions = {}
    probabilities = {}
    heldout_rows = []

    progress = st.progress(0, text="Training and evaluating models...")
    for i, model_name in enumerate(model_names):
        model = build_pipeline(model_name, k, int(random_state))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = positive_probability(model, X_test)

        fitted_models[model_name] = model
        predictions[model_name] = pred
        probabilities[model_name] = prob
        heldout_rows.append({
            "Model": model_name,
            **metric_row(y_test, pred, prob)
        })
        progress.progress((i + 1) / (len(model_names) + 1),
                          text=f"Completed {model_name}...")

    heldout = pd.DataFrame(heldout_rows).set_index("Model")

    # Five-fold cross-validation on the training partition only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
    cv_rows = []
    for i, model_name in enumerate(model_names):
        cv_model = build_pipeline(model_name, k, int(random_state))
        scores = cross_validate(
            cv_model,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "roc_auc_bipolar": bipolar_roc_auc_scorer},
            n_jobs=1,
            return_train_score=False
        )
        cv_rows.append({
            "Model": model_name,
            "CV Accuracy (mean)": scores["test_accuracy"].mean(),
            "CV Accuracy (SD)": scores["test_accuracy"].std(),
            "CV ROC-AUC (Bipolar mean)": scores["test_roc_auc_bipolar"].mean(),
            "CV ROC-AUC (Bipolar SD)": scores["test_roc_auc_bipolar"].std()
        })
        progress.progress((len(model_names) + i + 1) / (2 * len(model_names)),
                          text=f"Completed {model_name} cross-validation...")

    progress.empty()
    cv_results = pd.DataFrame(cv_rows).set_index("Model")

    st.session_state["dashboard_results"] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": fitted_models,
        "predictions": predictions,
        "probabilities": probabilities,
        "heldout": heldout,
        "cv_results": cv_results,
        "feature_count": k,
        "test_fraction": float(test_fraction),
        "random_state": int(random_state)
    }

results = st.session_state["dashboard_results"]

# ---------------------------------------------------------------------
# Results views
# ---------------------------------------------------------------------
st.divider()
st.header("Model performance")

tab1, tab2, tab3, tab4 = st.tabs([
    "Model comparison", "Detailed evaluation", "Molecular features", "New sample prediction"
])

with tab1:
    st.subheader("Held-out test-set comparison")
    st.caption(
        f"Stratified {int((1-results['test_fraction'])*100)}:{int(results['test_fraction']*100)} "
        f"train/test split · {len(results['y_train'])} train samples · "
        f"{len(results['y_test'])} test samples · {results['feature_count']} selected features"
    )
    st.dataframe(results["heldout"].style.format("{:.4f}"), use_container_width=True)

    st.subheader("Five-fold cross-validation on training data")
    st.dataframe(results["cv_results"].style.format("{:.4f}"), use_container_width=True)

    best_cv = results["cv_results"]["CV ROC-AUC (Bipolar mean)"].idxmax()
    st.info(
        f"Highest mean CV ROC-AUC: **{best_cv}** "
        f"({results['cv_results'].loc[best_cv, 'CV ROC-AUC (Bipolar mean)']:.4f}). "
        "This is model-selection evidence, not independent clinical validation."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    results["heldout"]["Accuracy"].plot(kind="bar", ax=ax, ylim=(0, 1), rot=0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Held-out test accuracy by model")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download model comparison CSV",
        results["heldout"].to_csv().encode("utf-8"),
        "heldout_test_model_comparison.csv",
        "text/csv"
    )
    st.download_button(
        "Download cross-validation CSV",
        results["cv_results"].to_csv().encode("utf-8"),
        "five_fold_cv_training_results.csv",
        "text/csv"
    )

with tab2:
    st.subheader(f"{detailed_model} — detailed evaluation")
    pred = results["predictions"][detailed_model]
    prob = results["probabilities"][detailed_model]
    y_test = results["y_test"]
    y_binary = (y_test == "Bipolar").astype(int)
    cm = confusion_matrix(y_test, pred, labels=["Bipolar", "Control"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        image = ax.imshow(cm, interpolation="nearest")
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Bipolar", "Control"],
            yticklabels=["Bipolar", "Control"],
            xlabel="Predicted label", ylabel="Actual label",
            title="Confusion matrix"
        )
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        st.markdown("**Classification report**")
        report = classification_report(
            y_test, pred, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y_binary, prob)
    auc_value = roc_auc_score(y_binary, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{detailed_model} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    pred_export = pd.DataFrame({
        "Sample": results["y_test"].index,
        "Actual_Diagnosis": results["y_test"].values,
        "Predicted_Diagnosis": pred,
        "Bipolar_probability": prob
    })
    st.download_button(
        "Download held-out test predictions",
        pred_export.to_csv(index=False).encode("utf-8"),
        f"{detailed_model.lower().replace(' ', '_')}_test_predictions.csv",
        "text/csv"
    )

with tab3:
    st.subheader("Top selected molecular features")
    model = results["models"][detailed_model]
    selector = model.named_steps["select"]
    selected_mask = selector.get_support()
    selected_genes = gene_names[selected_mask]
    scores = selector.scores_[selected_mask]
    features = pd.DataFrame({
        "Gene ID": selected_genes,
        "ANOVA F-score": scores
    }).replace([np.inf, -np.inf], np.nan).dropna()
    features = features.sort_values("ANOVA F-score", ascending=False).head(30)
    st.dataframe(features, use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_features = features.head(15).sort_values("ANOVA F-score")
    ax.barh(plot_features["Gene ID"], plot_features["ANOVA F-score"])
    ax.set_xlabel("ANOVA F-score")
    ax.set_title("Top selected features (statistical ranking)")
    st.pyplot(fig)
    plt.close(fig)
    st.warning(
        "These are statistically selected features from this dataset and split. "
        "They are not confirmed bipolar biomarkers; biological interpretation and independent validation are still required."
    )
    st.download_button(
        "Download selected features",
        features.to_csv(index=False).encode("utf-8"),
        "selected_molecular_features.csv",
        "text/csv"
    )

with tab4:
    st.subheader("Predict new samples from a compatible count matrix")
    st.write(
        "Upload a tab-separated count matrix with gene IDs in the first column and "
        "one or more new samples in the remaining columns. Gene IDs must overlap "
        "with the GSE124326 training features."
    )
    new_file = st.file_uploader(
        "Upload new RNA-seq-derived gene-count matrix",
        type=["txt", "tsv", "csv"],
        key="new_prediction_file"
    )
    if new_file is not None:
        try:
            if new_file.name.lower().endswith(".csv"):
                new_df = pd.read_csv(new_file)
            else:
                new_df = pd.read_csv(new_file, sep="\t", low_memory=False)

            gene_col = "gene" if "gene" in new_df.columns else new_df.columns[0]
            new_df[gene_col] = new_df[gene_col].astype(str)
            new_df = new_df.drop_duplicates(subset=[gene_col]).set_index(gene_col)
            new_df = new_df.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)

            # Restrict to the exact gene universe used for training.
            training_genes = list(gene_names)
            aligned = new_df.reindex(training_genes).fillna(0)
            new_counts = aligned.T
            library_sizes = new_counts.sum(axis=1).replace(0, np.nan)
            new_cpm = new_counts.div(library_sizes, axis=0) * 1_000_000
            new_log = np.log2(new_cpm.fillna(0) + 1).astype(np.float32)

            prediction_model = results["models"][detailed_model]
            new_pred = prediction_model.predict(new_log)
            new_prob = positive_probability(prediction_model, new_log)
            output = pd.DataFrame({
                "Sample": new_log.index,
                "Predicted class": new_pred,
                "Bipolar probability": new_prob
            })
            st.dataframe(output, use_container_width=True)
            st.download_button(
                "Download new-sample predictions",
                output.to_csv(index=False).encode("utf-8"),
                "new_sample_predictions.csv",
                "text/csv"
            )
        except Exception as exc:
            st.error(f"Could not process the uploaded matrix: {exc}")

st.divider()
st.caption(
    "Scientific limitation: this implementation analyzes processed gene-level RNA-seq counts, "
    "not raw FASTQ reads. Predictions are exploratory and must not be used for medical diagnosis."
)
