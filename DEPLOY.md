# Deploying the demo on Streamlit Community Cloud

This guide puts the interactive risk-screening demo (`app.py`) online for free so it
can be linked from a portfolio or a LinkedIn **Featured** section.

> ⚕️ Reminder: this is an **educational screening demo, not a medical device**. Keep
> the disclaimer visible wherever you share it.

---

## What makes this deploy work

`models/` and `reports/` are intentionally **git-ignored** — they are regenerated, not
source code. That raises one question: *if the trained model isn't in the repo, what
does the deployed app load?*

`app.py` handles this itself. On first boot it checks for
`models/diabetes_pipeline.joblib`; if it's missing, it **trains the model once**
(a few seconds on this 768-row dataset) and caches it for the life of the container
via `@st.cache_resource`. Training in the live environment also sidesteps any
scikit-learn / pickle version mismatch that shipping a pre-built `.joblib` could cause.

So there is **nothing extra to commit** to make the deploy work. Just point Community
Cloud at `app.py`.

---

## Prerequisites

- The repo is on GitHub: `Anubhavxd0/diabetes-ai-prediction`.
- `app.py` is at the repo root and `requirements.txt` lists the dependencies (both
  already true here).
- A free account at <https://streamlit.io/cloud> (sign in with GitHub).

---

## Step-by-step

1. **Merge or note the branch.** Deploy from whichever branch you want live —
   `main` after merging PR #1, or the `rebuild-diabetes-model` branch directly for a
   preview. You can pick the branch in the deploy form.

2. **Go to Community Cloud** → <https://share.streamlit.io> → sign in with GitHub and
   authorize access to the repository.

3. **Create the app** → click **New app** → **Deploy a public app from GitHub**, then
   fill in:
   - **Repository:** `Anubhavxd0/diabetes-ai-prediction`
   - **Branch:** `main` (or `rebuild-diabetes-model`)
   - **Main file path:** `app.py`

4. *(Optional)* **Advanced settings → Python version:** choose **3.11** to match the
   environment this project was verified on.

5. **Click Deploy.** The first build installs `requirements.txt`, then the app trains
   the model on first load (you'll briefly see *"Preparing model (first run only)…"*).
   Subsequent loads are instant.

6. **Grab the URL.** You'll get a public link like
   `https://<your-app-name>.streamlit.app`. That's the link for LinkedIn.

---

## Verifying it works

- The header should read **"🩺 Diabetes Risk Screening"** with a caption showing
  **Model in use: Logistic Regression**.
- Enter values (e.g. Glucose 148, BMI 33.6, Age 50) and click **Assess risk** — you
  should get a risk percentage, a progress bar, and a colour-coded band.
- Leaving a field at `0` (e.g. Insulin) is treated as "not measured" and imputed by
  the pipeline — this is intended behaviour, not a bug.

---

## Updating the live app

Community Cloud auto-redeploys on every push to the deployed branch. To ship changes:

```bash
git add <files>
git commit -m "…"
git push        # via your normal workflow
```

The app rebuilds automatically. If you ever change the training code, the cached model
is rebuilt on the next cold start.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| *"Could not prepare the model … dataset could not be downloaded"* | The dataset is fetched from a public GitHub mirror on first run. Community Cloud has internet, so this is usually a transient network blip — **reboot the app** from the app menu. |
| App stuck on *"Preparing model…"* on the very first load | Normal for the first few seconds while it trains. Only happens on a cold container. |
| `ModuleNotFoundError` | A dependency is missing from `requirements.txt`. Add it, commit, push. |
| Wrong Python version behaviour | Set Python to **3.11** in the app's Advanced settings and reboot. |

---

## Alternative: pre-train instead of train-on-first-run

If you'd rather not train on the server (e.g. to guarantee identical numbers to the
committed `reports/metrics.json`), you can commit the artifact instead:

```bash
python src/train.py
git add -f models/diabetes_pipeline.joblib
git commit -m "Ship pre-trained pipeline for deploy"
```

If you do this, **pin exact versions** in `requirements.txt` (especially
`scikit-learn`) so the deployed environment can unpickle the model without a version
warning. The train-on-first-run approach used by default avoids this entirely, which
is why it's the recommended path.
