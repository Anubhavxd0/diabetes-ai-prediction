# Diabetes Risk Prediction

A machine-learning model that estimates **diabetes risk from routine clinical lab
values**, built on the Pima Indians Diabetes dataset (768 patients, 8 features).

**Author:** Anubhav Rai — BSc Medical Laboratory Technology (MLT) + physical sciences
**Goal:** a small but *methodologically honest* clinical ML project, not just a high
accuracy number.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anubhavxd0/diabetes-ai-prediction/blob/main/diabetes_prediction.ipynb)

> ⚕️ **Disclaimer:** This is an educational project, **not a medical device**. It
> must not be used to diagnose, treat, or make decisions about any real person.

---

## Why this project is built the way it is

Most first-attempt diabetes notebooks quietly make the same mistakes. This project
fixes them on purpose, and each fix is a talking point in an interview:

| Common mistake | What this project does instead |
|----------------|-------------------------------|
| Fills missing values using the **whole** dataset before splitting → **data leakage** | Splits first, then imputes **inside a `Pipeline`** so fill values are learned from the training set only |
| Treats impossible `0`s (Glucose/BMI/BP = 0) as real measurements | Marks them as **missing** and imputes them properly |
| Reports **accuracy only** on imbalanced data (~35% positive) | Reports **precision, recall, F1 and ROC-AUC**, and prioritises **recall** (catching real diabetics) |
| Saves the wrong / bare model | Saves the **entire pipeline** (imputer + scaler + model) so inference matches training |
| One model, tuned by guessing | **Stratified 5-fold cross-validation** across 3 models, selected by ROC-AUC |

### Why recall matters here
For a screening tool, a **false negative** (telling a diabetic patient they are fine)
is more dangerous than a false alarm. That is why the models use
`class_weight="balanced"` and why ROC-AUC and recall — not raw accuracy — drive model
selection.

---

## Results

These are the **actual metrics from a full run** of `python src/train.py`
(scikit-learn 1.9, `random_state=42`, 80/20 stratified split). They are also written
to `reports/metrics.json` and are reproducible with the command above.

### Model comparison — stratified 5-fold cross-validation (training set)

Models were compared on the training folds only; the winner was selected by mean
cross-validated **ROC-AUC**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| **Logistic Regression** ✅ | 0.762 | 0.649 | 0.701 | 0.672 | **0.844** |
| Random Forest | 0.761 | 0.634 | 0.748 | 0.686 | 0.838 |
| Gradient Boosting | 0.756 | 0.672 | 0.598 | 0.631 | 0.821 |

**Winning model: Logistic Regression** — best mean CV ROC-AUC (0.844 ± 0.016). It also
happens to be the most interpretable of the three, which is a bonus for a clinical
screening context.

### Held-out test set (Logistic Regression, 154 patients never seen in training)

| Metric | Score |
|--------|:-----:|
| ROC-AUC | 0.813 |
| Accuracy | 0.734 |
| Precision (diabetic class) | 0.603 |
| **Recall (diabetic class)** | **0.704** |
| F1 (diabetic class) | 0.650 |

On the test set the model correctly flags about **70% of true diabetic cases** — the
recall we deliberately optimised for, since a missed case is the costliest error in
screening. The gap between the CV ROC-AUC (0.844) and the test ROC-AUC (0.813) is
small and expected for a dataset this size (768 rows), and is an honest reflection of
generalisation rather than a leak-inflated score.

Generated figures (in `reports/`): `confusion_matrix.png`, `roc_curve.png`,
`feature_importance.png`.

---

## Project structure

```
diabetes-ai-prediction/
├── diabetes_prediction.ipynb   # Clean, narrated notebook (open in Colab)
├── app.py                      # Streamlit interactive demo
├── requirements.txt
├── src/
│   ├── data.py                 # Load + cache dataset, mark impossible zeros as missing
│   ├── train.py                # Leak-free pipeline, model comparison, evaluation, saving
│   └── predict.py              # Single-patient inference
├── models/                     # (generated) saved pipeline
└── reports/                    # (generated) metrics.json + figures
```

## How to run

### Option A — Google Colab (easiest)
Click the **Open in Colab** badge above and run all cells top to bottom. Colab already
has all dependencies and internet access to fetch the dataset.

### Option B — Locally
```bash
pip install -r requirements.txt

# Train, evaluate, and save the model + reports
python src/train.py

# Predict for the example patient
python src/predict.py
```

### Option C — Interactive demo (Streamlit)
```bash
pip install -r requirements.txt
python src/train.py          # creates models/diabetes_pipeline.joblib
streamlit run app.py
```
The demo can be deployed for free on
[Streamlit Community Cloud](https://streamlit.io/cloud) and linked from a portfolio
or LinkedIn "Featured" section. See **[DEPLOY.md](DEPLOY.md)** for a step-by-step
guide. (No model file needs to be committed — the app trains itself once on first
boot.)

---

## The data

[Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
— 768 female patients of Pima Indian heritage, aged 21+.

| Feature | Meaning |
|---------|---------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigree | Family-history likelihood score |
| Age | Age (years) |
| **Outcome** | Target: 1 = diabetes, 0 = no diabetes |

---

## Limitations

- **Population bias:** female Pima Indian patients only — results may not transfer to
  men, children, or other populations (including Indian cohorts).
- **Small dataset:** 768 records.
- **Heavy imputation on Insulin:** ~half of insulin values were missing.
- **Not clinically validated.**

## Possible next steps

- Tune the decision threshold to maximise recall for screening.
- Add probability **calibration** and **SHAP** explanations.
- Validate on an Indian patient cohort.
