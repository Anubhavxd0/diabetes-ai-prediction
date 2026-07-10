"""Train and evaluate diabetes-risk models the right way.

What this script does (and why it is different from a naive first attempt):

1.  Treats impossible zeros (Glucose/BMI/etc. == 0) as MISSING.
2.  Splits the data BEFORE any imputation, using a *stratified* split so the
    train and test sets keep the same diabetic/non-diabetic ratio.
3.  Puts imputation + scaling inside a scikit-learn ``Pipeline`` so those steps
    are fit on training data only. This removes the data leakage that inflates
    scores in most beginner notebooks.
4.  Compares three models with *stratified 5-fold cross-validation* and reports
    accuracy, precision, recall, F1 and ROC-AUC -- not just accuracy.
5.  Selects the best model by ROC-AUC, evaluates it honestly on the held-out
    test set, and saves the full pipeline plus a metrics report and figures.

Run:  python src/train.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import FEATURES, load_dataset

RANDOM_STATE = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def build_pipeline(estimator) -> Pipeline:
    """Impute (median) -> scale -> classify, all fit on training data only."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def candidate_models() -> dict[str, object]:
    """Interpretable baseline + two ensembles.

    ``class_weight="balanced"`` nudges the models to pay more attention to the
    minority (diabetic) class -- important for a screening tool where a missed
    case (false negative) is worse than a false alarm.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def cross_validate_models(X_train, y_train) -> pd.DataFrame:
    """5-fold stratified CV; return a table of mean +/- std for each metric."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for name, estimator in candidate_models().items():
        pipe = build_pipeline(estimator)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def evaluate_on_test(pipe, X_test, y_test) -> dict:
    """Compute the full set of honest test-set metrics for the final model."""
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Diabetes", "Diabetes"], output_dict=True
        ),
    }


def save_figures(pipe, best_name, X_test, y_test) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, display_labels=["No Diabetes", "Diabetes"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title(f"ROC Curve - {best_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), dpi=120)
    plt.close(fig)

    # Feature importance (permutation importance works for any model type)
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(FEATURES)[order], result.importances_mean[order], color="#2c7fb8")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title(f"Feature Importance - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading dataset (impossible zeros marked as missing)...")
    X, y = load_dataset()
    print(f"  samples={len(X)}  features={X.shape[1]}  positive_rate={y.mean():.1%}")

    # Stratified split keeps the class balance identical in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\nCross-validating candidate models (stratified 5-fold)...")
    cv_table = cross_validate_models(X_train, y_train)
    display_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    print(cv_table[display_cols].round(3).to_string())

    # Pick the winner by mean cross-validated ROC-AUC.
    best_name = cv_table["roc_auc"].idxmax()
    print(f"\nBest model by CV ROC-AUC: {best_name}")

    best_pipe = build_pipeline(candidate_models()[best_name])
    best_pipe.fit(X_train, y_train)

    print("\nEvaluating the best model on the held-out test set...")
    test_metrics = evaluate_on_test(best_pipe, X_test, y_test)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:9s}: {test_metrics[k]:.3f}")

    save_figures(best_pipe, best_name, X_test, y_test)

    # Persist the full pipeline (imputer + scaler + model) so inference matches
    # training exactly. Never save just the bare estimator.
    model_path = os.path.join(MODELS_DIR, "diabetes_pipeline.joblib")
    dump({"pipeline": best_pipe, "features": FEATURES, "model_name": best_name}, model_path)

    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "cross_validation": cv_table.round(4).reset_index().to_dict("records"),
                "test_set": {k: test_metrics[k] for k in display_cols},
                "test_classification_report": test_metrics["classification_report"],
            },
            f,
            indent=2,
        )

    print(f"\nSaved model    -> {os.path.relpath(model_path, PROJECT_ROOT)}")
    print(f"Saved metrics  -> {os.path.relpath(metrics_path, PROJECT_ROOT)}")
    print(f"Saved figures  -> {os.path.relpath(REPORTS_DIR, PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
