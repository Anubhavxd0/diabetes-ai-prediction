"""Interactive diabetes-risk screening demo.

Run locally:
    streamlit run app.py

Or deploy for free on Streamlit Community Cloud so it can be linked from a
portfolio or personal site.

NOTE: This is an educational screening demo, not a diagnostic tool.
"""
import os
import sys

import streamlit as st

# Make the modules in src/ importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import MODEL_PATH, load_model, predict_patient  # noqa: E402

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Screening")
st.caption(
    "Machine-learning screening demo trained on the Pima Indians Diabetes dataset. "
    "Educational use only — not a medical diagnosis."
)


@st.cache_resource(show_spinner="Preparing model (first run only)…")
def get_artifact():
    """Load the trained pipeline, training it once if it is not present yet.

    On a fresh deploy (e.g. Streamlit Community Cloud) the ``models/`` directory
    is not committed to git, so no artifact exists on first boot. Rather than
    ask a visitor to run a script, we train the model once on startup — it takes
    only a few seconds on this small dataset — and cache it for the container's
    lifetime. Training in the live environment also avoids any pickle/scikit-learn
    version mismatch that shipping a pre-built ``.joblib`` could cause.
    """
    if not os.path.exists(MODEL_PATH):
        import train  # local module in src/ (added to sys.path above)

        train.main()
    return load_model()


try:
    artifact = get_artifact()
except Exception as exc:  # pragma: no cover - surfaced in the UI
    st.error(
        "Could not prepare the model. This usually means the dataset could not be "
        f"downloaded on first run.\n\nDetails: `{exc}`"
    )
    st.stop()

st.caption(f"Model in use: **{artifact.get('model_name', 'unknown')}**")

st.subheader("Enter patient measurements")
st.write("Leave a field at 0 if it was not measured — the model treats it as missing.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
    pedigree = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4, step=0.01
    )
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=33)

if st.button("Assess risk", type="primary"):
    patient = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigree": pedigree,
        "Age": age,
    }
    result = predict_patient(patient, artifact=artifact)
    prob = result["probability"]
    band = result["risk_band"]

    st.metric("Estimated diabetes risk", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if band == "High":
        st.error(f"**{band} risk.** Consider clinical follow-up and confirmatory testing.")
    elif band == "Moderate":
        st.warning(f"**{band} risk.** Lifestyle review and monitoring may be advisable.")
    else:
        st.success(f"**{band} risk.** No elevated risk indicated by this screening.")

st.divider()
st.caption(
    "Limitations: trained on female patients of Pima Indian heritage (768 records). "
    "Not validated for clinical use and not representative of all populations."
)
