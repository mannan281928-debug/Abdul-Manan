"""
Streamlit web app for the Heart Disease Prediction System (Step 10 of the notebook).

Run locally:   python -m streamlit run app.py

Needs in the same folder:
  svc_trained_model.pkl       (required, created by Step 6.4 of the notebook)
  heart_disease_combined.csv  (optional, the raw dataset; shown below the prediction)
  heart_logo.png, heart_icon.png (optional, the red heartbeat logo and the browser-tab icon)
  model_metrics.json          (optional, test-set accuracy, recall, precision, F1 and confusion matrix;
                               created by model_metrics_cell.py, shown at the bottom of the page)
  .streamlit/config.toml      (optional, dark theme)

Optional packages:  fpdf2  (enables the PDF download button:  pip install fpdf2)
"""

import base64
import inspect
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder

try:  # optional: only needed for the PDF download button
    from fpdf import FPDF
except ImportError:
    FPDF = None

# =============================================================================
# Configuration
# =============================================================================

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "svc_trained_model.pkl"
DATA_PATH = APP_DIR / "heart_disease_combined.csv"
LOGO_PATH = APP_DIR / "heart_logo.png"
ICON_PATH = APP_DIR / "heart_icon.png"
METRICS_PATH = APP_DIR / "model_metrics.json"

STUDENT_NAME = "Abdul Manan"
STUDENT_ID = "FA24-BSE-001"

# Fallback age range, used only when the dataset file is not available
DEFAULT_AGE_RANGE = (28, 77)

VERDICT_TEXT = {1: "Heart disease", 0: "No heart disease"}
BOUNDARY_NOTE = (
    "Close to the decision boundary: the verdict and the probability point in slightly different directions, "
    "so treat this result as uncertain"
)
SCORE_NOTE = (
    "This is an uncalibrated model score, not a true probability. "
    "Retrain with SVC(probability=True) to get a real risk percentage"
)
DISCLAIMER = (
    "Educational model built on four details. It cannot diagnose or rule out heart disease. "
    "Please see a doctor about any health concern."
)

# Length of the gauge's half circle (radius 80), used to fill it to the right percentage
GAUGE_ARC_LENGTH = 251.33

# Same fixed category lists the notebook used in Step 5.1, so the numeric codes match the trained model.
# Gender, Chest pain and Exercise angina need an encoder; Age is used as a plain number.
CHEST_PAIN_CODES = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
CATEGORIES = {
    "Gender": ["Male", "Female"],
    "ChestPain": list(CHEST_PAIN_CODES.values()),
    "ExerciseAngina": ["No", "Yes"],
}

# Session-state key -> dataset column
FIELDS = {"age": "Age", "gender": "Gender", "chest_pain": "ChestPain", "exercise_angina": "ExerciseAngina"}

EXERCISE_ANGINA_NOTES = {
    "Yes": "Exercise-induced angina: chest pain brought on by exercise",
    "No": "No chest pain brought on by exercise",
}
CHEST_PAIN_NOTES = {
    "Typical angina": "Chest pain brought on by effort or stress, eased by rest",
    "Atypical angina": "Chest pain with some, but not all, typical angina features",
    "Non-anginal pain": "Chest pain that is not caused by the heart",
    "Asymptomatic": "No chest pain symptoms",
}

# Newer Streamlit uses width="stretch"; older versions use use_container_width=True
try:
    STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}
except (TypeError, ValueError):
    STRETCH = {"use_container_width": True}


# =============================================================================
# Styling
# =============================================================================

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Figtree:wght@400;500;600;700&display=swap');

:root {
  --hd-bg: #070b14;
  --hd-panel: rgba(255, 255, 255, 0.045);
  --hd-panel-strong: rgba(255, 255, 255, 0.07);
  --hd-line: rgba(255, 255, 255, 0.10);
  --hd-text: #e8ecf5;
  --hd-muted: #9aa4b8;
  --hd-heart: #ff4d5a;
  --hd-clear: #22d3b6;
  --hd-accent: #6c8cff;
}

/* ---------- base ---------- */
.stApp {
  background:
    radial-gradient(900px 520px at 12% -8%, rgba(108, 140, 255, 0.20), transparent 60%),
    radial-gradient(800px 520px at 92% 0%, rgba(255, 77, 90, 0.16), transparent 60%),
    radial-gradient(700px 500px at 50% 110%, rgba(34, 211, 182, 0.10), transparent 60%),
    var(--hd-bg);
  color: var(--hd-text);
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button { font-family: 'Figtree', system-ui, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 4.5rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
.stApp [data-testid="stCaptionContainer"], .stApp .stCaption { color: var(--hd-muted); }

/* ---------- student badge (top left) ---------- */
.hd-student {
  position: fixed; top: 0.9rem; left: 1.2rem; z-index: 1000;
  display: flex; align-items: center; gap: 0.65rem;
  font-size: 0.85rem; line-height: 1.3; padding: 0.45rem 0.9rem 0.45rem 0.75rem;
  border: 1px solid var(--hd-line); border-radius: 12px;
  background: rgba(15, 22, 38, 0.72); backdrop-filter: blur(10px);
  box-shadow: 0 0 18px rgba(108, 140, 255, 0.18);
}
.hd-student-dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--hd-clear);
  box-shadow: 0 0 10px var(--hd-clear); animation: hd-blink 2.4s ease-in-out infinite;
}
.hd-student b { font-weight: 600; display: block; }
.hd-student span { color: var(--hd-muted); font-size: 0.78rem; }

/* ---------- hero ---------- */
.hd-hero { text-align: center; margin-bottom: 1.4rem; }
.hd-logo {
  display: block; width: min(420px, 85%); height: auto; margin: 0 auto 0.5rem auto;
  filter: drop-shadow(0 0 14px rgba(255, 77, 90, 0.55));
  animation: hd-logo-glow 3.2s ease-in-out infinite;
}
.hd-hero h1 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600; letter-spacing: -0.02em;
  font-size: clamp(2rem, 5vw, 3rem); line-height: 1.05; margin: 0; padding: 0;
  background: linear-gradient(90deg, #ffffff 0%, #ffb3b8 45%, var(--hd-heart) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  filter: drop-shadow(0 0 18px rgba(255, 77, 90, 0.35));
}
.hd-hero p { font-size: 1.05rem; color: var(--hd-muted); margin: 0.5rem auto 0 auto; max-width: 52ch; }
.hd-divider {
  height: 1px; width: min(320px, 60%); margin: 1.1rem auto 0 auto;
  background: linear-gradient(90deg, transparent, var(--hd-heart), transparent);
  box-shadow: 0 0 12px rgba(255, 77, 90, 0.6);
}

.hd-section {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.45rem;
  margin: 2.2rem 0 0.4rem 0; text-shadow: 0 0 16px rgba(108, 140, 255, 0.35);
}

/* ---------- input boxes ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 16px;
  border: 1px solid var(--hd-line) !important;
  background: var(--hd-panel);
  transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stVerticalBlockBorderWrapper"]:focus-within {
  border-color: rgba(108, 140, 255, 0.65) !important;
  box-shadow: 0 0 0 1px rgba(108, 140, 255, 0.25), 0 0 26px rgba(108, 140, 255, 0.28);
  transform: translateY(-2px);
}
.hd-box-title { font-weight: 700; font-size: 0.98rem; margin: 0 0 0.5rem 0; letter-spacing: 0.01em; }
.hd-box-hint { font-size: 0.8rem; color: var(--hd-muted); margin: 0.45rem 0 0 0; min-height: 2.4em; }
.hd-small.is-warn { color: #ffc46b; }

div[data-baseweb="select"] > div {
  background: rgba(255, 255, 255, 0.05); border-color: var(--hd-line); border-radius: 10px;
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-baseweb="select"]:focus-within > div {
  border-color: var(--hd-accent); box-shadow: 0 0 14px rgba(108, 140, 255, 0.55);
}
div[data-baseweb="popover"] ul { background: #0f1626; }

/* ---------- toggle ---------- */
.stApp [data-testid="stToggle"] label p, .stApp [data-testid="stCheckbox"] label p { color: var(--hd-text); }

/* ---------- predict button ---------- */
.stButton > button {
  border: none; border-radius: 12px; font-weight: 700; letter-spacing: 0.03em;
  padding: 0.7rem 1rem; color: #ffffff;
  background: linear-gradient(135deg, #ff4d5a 0%, #ff7a59 100%);
  box-shadow: 0 0 20px rgba(255, 77, 90, 0.50), 0 0 50px rgba(255, 77, 90, 0.22);
  transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}
.stButton > button:hover {
  transform: translateY(-2px); filter: brightness(1.08);
  box-shadow: 0 0 28px rgba(255, 77, 90, 0.75), 0 0 70px rgba(255, 77, 90, 0.35);
  color: #ffffff;
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus-visible { outline: 2px solid rgba(255, 255, 255, 0.7); outline-offset: 2px; }

/* ---------- result card ---------- */
.hd-card {
  position: relative; border: 1px solid var(--hd-line); border-radius: 18px;
  padding: 1.6rem 1.7rem; margin-top: 1.2rem;
  background: linear-gradient(160deg, var(--hd-panel-strong), var(--hd-panel));
  backdrop-filter: blur(8px);
}
.hd-card.is-heart { border-color: rgba(255, 77, 90, 0.55); animation: hd-pulse-heart 3s ease-in-out infinite; }
.hd-card.is-clear { border-color: rgba(34, 211, 182, 0.55); animation: hd-pulse-clear 3s ease-in-out infinite; }
.hd-status {
  display: inline-block; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.25rem 0.7rem; border-radius: 999px; margin: 0 0 0.7rem 0;
  border: 1px solid var(--hd-line); color: var(--hd-muted); background: rgba(255, 255, 255, 0.04);
}
.hd-verdict {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.3rem; line-height: 1.1; margin: 0 0 1.1rem 0;
}
.is-heart .hd-verdict { color: var(--hd-heart); text-shadow: 0 0 22px rgba(255, 77, 90, 0.65); }
.is-clear .hd-verdict { color: var(--hd-clear); text-shadow: 0 0 22px rgba(34, 211, 182, 0.65); }

.hd-recap { width: 100%; border-collapse: collapse; font-size: 0.94rem; margin: 0 0 1rem 0; }
.hd-recap td { padding: 0.5rem 0; border-top: 1px solid var(--hd-line); }
.hd-recap td:first-child { color: var(--hd-muted); width: 48%; }
.hd-recap td:last-child { font-weight: 600; }

.hd-small { font-size: 0.82rem; color: var(--hd-muted); margin: 0.35rem 0 0 0; }
.hd-note { font-size: 0.85rem; color: var(--hd-muted); margin: 1rem 0 0 0; text-align: center; }
.hd-footer { text-align: center; font-size: 0.78rem; color: var(--hd-muted); margin-top: 2.5rem; opacity: 0.8; }

/* ---------- result layout, gauge and confidence bar ---------- */
.hd-result-grid { display: grid; grid-template-columns: 1fr minmax(210px, 270px); gap: 1.8rem; align-items: center; }
.hd-result-grid.is-single { grid-template-columns: 1fr; }
.hd-gauge { display: block; width: 100%; height: auto; }
.hd-gauge-track { stroke: rgba(255, 255, 255, 0.10); }
.hd-gauge-fill { stroke: var(--hd-accent); }
.is-heart .hd-gauge-fill { stroke: var(--hd-heart); filter: drop-shadow(0 0 7px rgba(255, 77, 90, 0.85)); }
.is-clear .hd-gauge-fill { stroke: var(--hd-clear); filter: drop-shadow(0 0 7px rgba(34, 211, 182, 0.85)); }
.hd-gauge-value { fill: var(--hd-text); font-family: 'Fraunces', Georgia, serif; font-size: 30px; font-weight: 600; }
.hd-gauge-label { fill: var(--hd-muted); font-family: 'Figtree', system-ui, sans-serif; font-size: 8.5px; letter-spacing: 0.1em; text-transform: uppercase; }

.hd-conf { margin-top: 0.7rem; }
.hd-conf-head { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--hd-muted); margin-bottom: 0.35rem; }
.hd-conf-head b { color: var(--hd-text); }
.hd-bar { display: block; width: 100%; height: 8px; border-radius: 999px; overflow: hidden; }
.hd-bar-track { fill: rgba(255, 255, 255, 0.10); }
.hd-bar-fill { fill: var(--hd-accent); }
.is-heart .hd-bar-fill { fill: var(--hd-heart); }
.is-clear .hd-bar-fill { fill: var(--hd-clear); }
.is-heart .hd-bar { filter: drop-shadow(0 0 5px rgba(255, 77, 90, 0.6)); }
.is-clear .hd-bar { filter: drop-shadow(0 0 5px rgba(34, 211, 182, 0.6)); }

/* ---------- download buttons ---------- */
[data-testid="stDownloadButton"] button {
  border: 1px solid rgba(108, 140, 255, 0.60); border-radius: 12px; font-weight: 600;
  background: rgba(108, 140, 255, 0.08); color: var(--hd-text);
  box-shadow: 0 0 14px rgba(108, 140, 255, 0.25);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
[data-testid="stDownloadButton"] button:hover {
  transform: translateY(-2px); border-color: var(--hd-accent); color: #ffffff;
  box-shadow: 0 0 24px rgba(108, 140, 255, 0.55);
}

/* ---------- model performance ---------- */
.hd-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-top: 0.8rem; }
.hd-metric { border: 1px solid var(--hd-line); border-radius: 14px; background: var(--hd-panel); padding: 0.9rem 1rem; }
.hd-metric span { display: block; font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--hd-muted); }
.hd-metric b { font-family: 'Fraunces', Georgia, serif; font-size: 1.8rem; font-weight: 600; }
.hd-metric.is-key { border-color: rgba(255, 77, 90, 0.55); box-shadow: 0 0 22px rgba(255, 77, 90, 0.25); }
.hd-metric.is-key b { color: var(--hd-heart); text-shadow: 0 0 16px rgba(255, 77, 90, 0.55); }
.hd-matrix { border-collapse: collapse; font-size: 0.9rem; margin: 1rem 0 0.4rem 0; }
.hd-matrix th, .hd-matrix td { border: 1px solid var(--hd-line); padding: 0.45rem 0.9rem; text-align: center; }
.hd-matrix th { color: var(--hd-muted); font-weight: 500; }
.hd-matrix td { font-weight: 700; }
@media (max-width: 720px) {
  .hd-result-grid { grid-template-columns: 1fr; }
  .hd-metrics { grid-template-columns: repeat(2, 1fr); }
}

/* ---------- animations ---------- */
@keyframes hd-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@keyframes hd-logo-glow {
  0%, 100% { filter: drop-shadow(0 0 10px rgba(255, 77, 90, 0.40)); }
  50%      { filter: drop-shadow(0 0 22px rgba(255, 77, 90, 0.85)); }
}
@keyframes hd-pulse-heart {
  0%, 100% { box-shadow: 0 0 18px rgba(255, 77, 90, 0.25), 0 0 60px rgba(255, 77, 90, 0.12); }
  50%      { box-shadow: 0 0 30px rgba(255, 77, 90, 0.50), 0 0 90px rgba(255, 77, 90, 0.22); }
}
@keyframes hd-pulse-clear {
  0%, 100% { box-shadow: 0 0 18px rgba(34, 211, 182, 0.25), 0 0 60px rgba(34, 211, 182, 0.12); }
  50%      { box-shadow: 0 0 30px rgba(34, 211, 182, 0.50), 0 0 90px rgba(34, 211, 182, 0.22); }
}
@media (prefers-reduced-motion: reduce) {
  .hd-logo, .hd-card, .hd-student-dot { animation: none !important; }
}
</style>
"""


# =============================================================================
# Data and model loading
# =============================================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_resource
def build_encoders():
    return {name: LabelEncoder().fit(values) for name, values in CATEGORIES.items()}


@st.cache_data
def load_dataset():
    """Optional: the dataset (cleaned like Step 3 and 4 of the notebook), with the real measurements."""
    if not DATA_PATH.exists():
        return None
    raw = pd.read_csv(DATA_PATH).replace("?", np.nan)
    for column in ["age", "sex", "cp", "exang", "num"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["age", "sex", "cp", "exang"]).reset_index(drop=True)
    return pd.DataFrame({
        "Age": raw["age"].astype(int),
        "Gender": raw["sex"].map({1.0: "Male", 0.0: "Female"}),
        "ChestPain": raw["cp"].astype(int).map(CHEST_PAIN_CODES),
        "ExerciseAngina": raw["exang"].map({1.0: "Yes", 0.0: "No"}),
        "HeartDisease": np.where(raw["num"] > 0, "Yes", "No"),
    })


@st.cache_data
def load_logo_html():
    """The red heartbeat logo, embedded in the page so it can sit right above the app name."""
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return f'<img class="hd-logo" src="data:image/png;base64,{encoded}" alt="Heart disease prediction logo">'


# =============================================================================
# Helpers
# =============================================================================

def exercise_angina_note(value):
    """Short note shown under the exercise-induced angina box."""
    return EXERCISE_ANGINA_NOTES.get(value, "Does exercise bring on chest pain (angina)?")


def chest_pain_note(value):
    """Short note shown under the chest pain box."""
    return CHEST_PAIN_NOTES.get(value, "Type of chest pain the patient reports")


def age_bounds(dataset):
    """Lowest and highest age: taken from the dataset when available, otherwise the default range."""
    if dataset is not None:
        return int(dataset["Age"].min()), int(dataset["Age"].max())
    return DEFAULT_AGE_RANGE


def full_range(column, dataset):
    """Every valid value for one column."""
    if column == "Age":
        low, high = age_bounds(dataset)
        return list(range(low, high + 1))
    return CATEGORIES[column]


def options_for(key, dataset):
    """Values for one box (full range, or only dataset-matching values when the toggle is on)."""
    column = FIELDS[key]
    full = full_range(column, dataset)
    linked = dataset is not None and st.session_state.get("dataset_mode", False)
    if not linked:
        return full

    mask = pd.Series(True, index=dataset.index)
    for other_key, other_column in FIELDS.items():
        chosen_value = st.session_state.get(other_key)
        if other_key != key and chosen_value is not None:
            mask &= dataset[other_column] == chosen_value
    found = set(dataset.loc[mask, column].tolist())
    return [value for value in full if value in found]


def clear_invalid_choices(dataset):
    """A choice that no longer fits the other boxes is cleared before the boxes are drawn."""
    for _ in range(len(FIELDS)):
        cleared = False
        for key in FIELDS:
            value = st.session_state.get(key)
            if value is not None and value not in options_for(key, dataset):
                st.session_state[key] = None
                cleared = True
        if not cleared:
            break


# =============================================================================
# Risk, metrics and exports
# =============================================================================

def risk_score(model, features):
    """Risk of heart disease between 0 and 1, plus its kind.

    "probability": a real probability from predict_proba (needs SVC(probability=True)).
    "score":       fallback for models saved without probabilities; the decision function squashed to 0..1,
                   NOT a calibrated probability.
    "none":        the model offers neither.
    """
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            return float(probabilities[classes.index(1)]), "probability"
        except (AttributeError, NotImplementedError, ValueError):
            pass
    if hasattr(model, "decision_function"):
        raw_score = float(np.ravel(model.decision_function(features))[0])
        return float(1.0 / (1.0 + np.exp(-raw_score))), "score"
    return None, "none"


def verdict_confidence(result):
    """Model's estimated probability for the verdict shown, or None when there is no real probability."""
    if result["risk_kind"] != "probability":
        return None
    return result["risk"] if result["prediction"] == 1 else 1.0 - result["risk"]


def is_near_boundary(result):
    """True when predict() and predict_proba() disagree (possible with SVC probabilities near 50%)."""
    if result["risk_kind"] != "probability":
        return False
    return (result["risk"] >= 0.5) != (result["prediction"] == 1)


def load_metrics():
    """Optional test-set metrics saved by model_metrics_cell.py; None when missing or unreadable."""
    if not METRICS_PATH.exists():
        return None
    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    required = ("accuracy", "precision", "recall", "f1")
    if not isinstance(data, dict) or not all(isinstance(data.get(key), (int, float)) for key in required):
        return None
    return data


def accuracy_phrase(metrics):
    """Sentence fragment about test performance for the note under the result."""
    if metrics is None:
        return "about 80% accurate on test data"
    return f"{metrics['accuracy']:.0%} accurate and {metrics['recall']:.0%} recall on test data"


def gauge_html(risk, kind):
    """Half-circle gauge filled to the risk percentage."""
    fraction = max(0.0, min(1.0, risk))
    label = "Heart disease risk" if kind == "probability" else "Model score"
    return (
        f'<svg class="hd-gauge" viewBox="0 0 200 118" role="img" aria-label="{label} {fraction:.0%}">'
        '<path class="hd-gauge-track" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14" stroke-linecap="round"/>'
        f'<path class="hd-gauge-fill" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14" '
        f'stroke-linecap="round" stroke-dasharray="{fraction * GAUGE_ARC_LENGTH:.1f} 400"/>'
        f'<text class="hd-gauge-value" x="100" y="86" text-anchor="middle">{fraction:.0%}</text>'
        f'<text class="hd-gauge-label" x="100" y="108" text-anchor="middle">{label}</text>'
        "</svg>"
    )


def confidence_html(confidence):
    """Thin bar showing how sure the model is about the verdict it gave."""
    fraction = max(0.0, min(1.0, confidence))
    return (
        '<div class="hd-conf">'
        f'<div class="hd-conf-head"><span>Confidence in this verdict</span><b>{fraction:.0%}</b></div>'
        f'<svg class="hd-bar" viewBox="0 0 100 6" preserveAspectRatio="none" role="img" '
        f'aria-label="Confidence {fraction:.0%}">'
        '<rect class="hd-bar-track" width="100" height="6"/>'
        f'<rect class="hd-bar-fill" width="{fraction * 100:.1f}" height="6"/>'
        "</svg></div>"
    )


def build_csv(result):
    """One-row CSV of this prediction."""
    profile = result["profile"]
    risk = result["risk"]
    row = {
        "timestamp": result["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "age": profile["Age"],
        "gender": profile["Gender"],
        "chest_pain": profile["ChestPain"],
        "exercise_angina": profile["ExerciseAngina"],
        "prediction": VERDICT_TEXT[result["prediction"]],
        "risk_percent": round(risk * 100, 1) if risk is not None else "",
        "risk_type": result["risk_kind"],
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def pdf_safe(text):
    """fpdf's built-in fonts only support Latin-1, so anything else is replaced."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(result, metrics=None):
    """One-page PDF report of this prediction (needs the optional fpdf2 package)."""
    profile = result["profile"]
    risk, risk_kind = result["risk"], result["risk_kind"]
    has_disease = result["prediction"] == 1
    accent = (220, 38, 38) if has_disease else (13, 148, 136)
    next_line = {"new_x": "LMARGIN", "new_y": "NEXT"}

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(25, 25, 25)
    pdf.cell(0, 12, "Heart Disease Prediction Report", **next_line)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, f"Generated {result['timestamp']:%Y-%m-%d %H:%M}", **next_line)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*accent)
    pdf.cell(0, 14, VERDICT_TEXT[result["prediction"]], **next_line)
    pdf.ln(2)

    if risk is not None:
        label = "Heart disease risk" if risk_kind == "probability" else "Model score (uncalibrated)"
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, f"{label}: {risk:.0%}", **next_line)
        bar_x, bar_y, bar_width = pdf.get_x(), pdf.get_y(), 120
        pdf.set_fill_color(225, 228, 235)
        pdf.rect(bar_x, bar_y, bar_width, 5, style="F")
        pdf.set_fill_color(*accent)
        pdf.rect(bar_x, bar_y, max(bar_width * risk, 0.1), 5, style="F")
        pdf.ln(9)
        confidence = verdict_confidence(result)
        if confidence is not None:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(90, 90, 90)
            pdf.cell(0, 6, f"Confidence in this verdict: {confidence:.0%}", **next_line)
        pdf.ln(4)

    rows = [
        ("Age", f"{profile['Age']} years"),
        ("Gender", profile["Gender"]),
        ("Chest pain", profile["ChestPain"]),
        ("Exercise angina", profile["ExerciseAngina"]),
    ]
    pdf.set_font("Helvetica", "", 11)
    for label, value in rows:
        pdf.set_text_color(110, 110, 110)
        pdf.cell(60, 8, pdf_safe(label), border="B")
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, pdf_safe(value), border="B", **next_line)
    pdf.ln(6)

    if metrics is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, "Model performance on test data", **next_line)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 90, 90)
        summary = (
            f"Accuracy {metrics['accuracy']:.0%}  |  Recall {metrics['recall']:.0%}  |  "
            f"Precision {metrics['precision']:.0%}  |  F1 {metrics['f1']:.0%}"
        )
        pdf.cell(0, 6, summary, **next_line)
        pdf.ln(4)

    cautions = []
    if is_near_boundary(result):
        cautions.append(BOUNDARY_NOTE + ".")
    if risk_kind == "score":
        cautions.append(SCORE_NOTE + ".")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(160, 90, 0)
    for caution in cautions:
        pdf.multi_cell(0, 5.5, pdf_safe(caution), **next_line)
        pdf.ln(1)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, pdf_safe(DISCLAIMER), **next_line)
    return bytes(pdf.output())


# =============================================================================
# Prediction
# =============================================================================

def run_prediction(model, encoders, dataset):
    """Runs when Predict is clicked: store the prediction, then clear the boxes for the next person."""
    values = {key: st.session_state.get(key) for key in FIELDS}
    done = sum(value is not None for value in values.values())
    if done < len(FIELDS):
        st.session_state["result"] = None
        st.session_state["warning"] = f"Please fill in all four details first. {done} of 4 done."
        return

    profile = {
        "Age": int(values["age"]),
        "Gender": values["gender"],
        "ChestPain": values["chest_pain"],
        "ExerciseAngina": values["exercise_angina"],
    }
    # Age goes in as a number; Gender, Chest pain and Exercise angina are label-encoded as in Step 8.3 of the notebook.
    # (Scaling is done inside the saved model, so nothing else is needed here.)
    user_input = pd.DataFrame({column: [value] for column, value in profile.items()})
    encoded = user_input.copy()
    for column in CATEGORIES:
        encoded[column] = encoders[column].transform(user_input[column])

    actual = None
    if dataset is not None and st.session_state.get("dataset_mode"):
        match = dataset[
            (dataset["Age"] == profile["Age"])
            & (dataset["Gender"] == profile["Gender"])
            & (dataset["ChestPain"] == profile["ChestPain"])
            & (dataset["ExerciseAngina"] == profile["ExerciseAngina"])
        ]
        if len(match) > 0:
            actual = (len(match), int((match["HeartDisease"] == "Yes").sum()))
    risk, risk_kind = risk_score(model, encoded)
    st.session_state["result"] = {
        "profile": profile,
        "prediction": int(model.predict(encoded)[0]),
        "risk": risk,
        "risk_kind": risk_kind,
        "encoded": encoded,
        "actual": actual,
        "timestamp": datetime.now(),
    }
    st.session_state["warning"] = None

    # Clear every box so the next person can be checked straight away
    for key in FIELDS:
        st.session_state[key] = None


# =============================================================================
# Page sections
# =============================================================================

def box_head(title):
    st.html(f'<p class="hd-box-title">{title}</p>')


def render_header():
    st.html(STYLE)
    st.html(
        '<div class="hd-student"><div class="hd-student-dot"></div>'
        f"<div><b>{STUDENT_NAME}</b><span>{STUDENT_ID}</span></div></div>"
    )
    st.html(
        '<div class="hd-hero">'
        f"{load_logo_html()}"
        "<h1>Heart disease prediction</h1>"
        "<p>Pick a value in each box, then press Predict.</p>"
        '<div class="hd-divider"></div>'
        "</div>"
    )


def render_inputs(dataset):
    st.toggle(
        "Only offer patients from the dataset (shows the real outcome after Predict)",
        key="dataset_mode",
    )

    low, high = age_bounds(dataset)
    col_age, col_gender, col_cp, col_exang = st.columns(4, gap="medium")

    with col_age:
        with st.container(border=True):
            box_head("Age")
            st.selectbox(
                "Age", options_for("age", dataset), index=None, placeholder="Click to choose",
                key="age", label_visibility="collapsed",
            )
            st.html(f'<p class="hd-box-hint">Whole years ({low} to {high} in the dataset)</p>')

    with col_gender:
        with st.container(border=True):
            box_head("Gender")
            st.selectbox(
                "Gender", options_for("gender", dataset), index=None, placeholder="Click to choose",
                key="gender", label_visibility="collapsed",
            )
            st.html('<p class="hd-box-hint">Male or female</p>')

    with col_cp:
        with st.container(border=True):
            box_head("Chest pain")
            chest_pain = st.selectbox(
                "Chest pain", options_for("chest_pain", dataset), index=None, placeholder="Click to choose",
                key="chest_pain", label_visibility="collapsed",
            )
            st.html(f'<p class="hd-box-hint">{chest_pain_note(chest_pain)}</p>')

    with col_exang:
        with st.container(border=True):
            box_head("Exercise angina")
            exercise_angina = st.selectbox(
                "Exercise angina", options_for("exercise_angina", dataset), index=None,
                placeholder="Click to choose", key="exercise_angina", label_visibility="collapsed",
            )
            st.html(f'<p class="hd-box-hint">{exercise_angina_note(exercise_angina)}</p>')

    if st.session_state.get("dataset_mode") and dataset is not None:
        st.caption("Each box only offers values that exist in the dataset together with your other choices.")
    else:
        st.caption("Any combination can be chosen. Values stay within the range of the dataset.")


def render_predict_button(model, encoders, dataset):
    _, button_col, _ = st.columns([1, 1, 1])
    with button_col:
        st.button(
            "Predict", type="primary", on_click=run_prediction,
            args=(model, encoders, dataset), **STRETCH,
        )


def render_result(result, metrics=None):
    profile, prediction, actual = result["profile"], result["prediction"], result["actual"]
    risk, risk_kind = result["risk"], result["risk_kind"]
    has_disease = prediction == 1
    css_class = "is-heart" if has_disease else "is-clear"
    verdict = VERDICT_TEXT[prediction]

    recap_rows = [
        ("Age", f'{profile["Age"]} years'),
        ("Gender", profile["Gender"]),
        ("Chest pain", profile["ChestPain"]),
        ("Exercise angina", profile["ExerciseAngina"]),
    ]
    if risk_kind == "probability":
        recap_rows.append(("Heart disease risk", f"{risk:.0%}"))
    note = ""
    if actual is not None:
        matches, positives = actual
        if positives in (0, matches):
            real_has_disease = positives == matches
            real_text = "Heart disease" if real_has_disease else "No heart disease"
            if matches > 1:
                real_text += f" (all {matches} matching patients)"
            verdict_text = "Correct" if real_has_disease == has_disease else "Wrong"
            recap_rows += [("Real outcome in dataset", real_text), ("Model was", verdict_text)]
        else:
            recap_rows.append(("Real outcome in dataset", f"{positives} of {matches} matching patients had heart disease"))
        note = (
            '<p class="hd-small">This patient comes from the dataset, so the model has probably seen them during training. '
            "A correct answer here does not prove it works on new people.</p>"
        )
    if is_near_boundary(result):
        note += f'<p class="hd-small is-warn">{BOUNDARY_NOTE}.</p>'
    if risk_kind == "score":
        note += f'<p class="hd-small">{SCORE_NOTE}.</p>'
    recap = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in recap_rows)

    side = ""
    if risk is not None:
        side = gauge_html(risk, risk_kind)
        confidence = verdict_confidence(result)
        if confidence is not None:
            side += confidence_html(confidence)
    grid_class = "hd-result-grid" if side else "hd-result-grid is-single"

    st.html(
        f'<div class="hd-card {css_class}">'
        '<p class="hd-status">Model prediction</p>'
        f'<p class="hd-verdict">{verdict}</p>'
        f'<div class="{grid_class}">'
        f'<div><table class="hd-recap">{recap}</table>{note}</div>'
        f"{f'<div>{side}</div>' if side else ''}"
        "</div>"
        "</div>"
        f'<p class="hd-note">Educational model built on four details ({accuracy_phrase(metrics)}). It cannot diagnose or '
        "rule out heart disease. Please see a doctor about any health concern.</p>"
    )


def render_downloads(result, metrics=None):
    """CSV and PDF download buttons for the prediction currently on screen."""
    stamp = result["timestamp"].strftime("%Y%m%d_%H%M%S")
    csv_col, pdf_col, _ = st.columns([1, 1, 2], gap="small")
    with csv_col:
        st.download_button(
            "Download CSV", data=build_csv(result), file_name=f"heart_prediction_{stamp}.csv",
            mime="text/csv", key="download_csv", **STRETCH,
        )
    with pdf_col:
        if FPDF is None:
            st.caption("PDF export needs fpdf2: pip install fpdf2")
            return
        try:
            pdf_bytes = build_pdf(result, metrics)
        except Exception as error:  # a broken PDF must never take the whole page down
            st.caption(f"PDF could not be created ({type(error).__name__}).")
            return
        st.download_button(
            "Download PDF", data=pdf_bytes, file_name=f"heart_prediction_{stamp}.pdf",
            mime="application/pdf", key="download_pdf", **STRETCH,
        )


def render_metrics(metrics):
    """Test-set performance panel; recall is highlighted because a missed patient is the costly mistake."""
    if metrics is None:
        return

    tiles = [("Accuracy", "accuracy", False), ("Recall", "recall", True),
             ("Precision", "precision", False), ("F1 score", "f1", False)]
    tile_html = "".join(
        f'<div class="hd-metric{" is-key" if key else ""}"><span>{label}</span><b>{metrics[name]:.0%}</b></div>'
        for label, name, key in tiles
    )

    matrix_html = ""
    matrix = metrics.get("confusion_matrix")
    if (
        isinstance(matrix, list) and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 and all(isinstance(v, int) for v in row) for row in matrix)
    ):
        (tn, fp), (fn, tp) = matrix
        matrix_html = (
            '<table class="hd-matrix">'
            "<tr><th></th><th>Predicted: No disease</th><th>Predicted: Disease</th></tr>"
            f"<tr><th>Actual: No disease</th><td>{tn}</td><td>{fp}</td></tr>"
            f"<tr><th>Actual: Disease</th><td>{fn}</td><td>{tp}</td></tr>"
            "</table>"
        )

    count = metrics.get("n_test")
    count_text = f" Measured on {count} test patients." if isinstance(count, int) else ""
    st.html(
        '<p class="hd-section">Model performance on test data</p>'
        f'<div class="hd-metrics">{tile_html}</div>'
        f"{matrix_html}"
        '<p class="hd-small">Recall is the share of real heart-disease patients the model catches. Missing a sick patient '
        f"is the costly mistake, so recall matters as much as accuracy.{count_text}</p>"
    )


def render_output(metrics=None):
    """Prediction shown below the button (stays on screen until the next Predict click)."""
    if st.session_state.get("warning"):
        st.warning(st.session_state["warning"])
        return

    result = st.session_state.get("result")
    if not isinstance(result, dict):  # nothing yet, or a result stored by an older version of this app
        st.session_state["result"] = None
        return

    render_result(result, metrics)
    render_downloads(result, metrics)

    st.html('<p class="hd-section">Encoded feature vector sent to the model</p>')
    st.dataframe(result["encoded"], hide_index=True, **STRETCH)
    st.caption(
        "Gender, Chest pain and Exercise angina are label-encoded (Gender: Female = 0, Male = 1; "
        "Chest pain: Asymptomatic = 0, Atypical angina = 1, Non-anginal pain = 2, Typical angina = 3; "
        "Exercise angina: No = 0, Yes = 1). Age stays a number; the model scales it itself."
    )


# =============================================================================
# Page
# =============================================================================

st.set_page_config(
    page_title="Heart disease prediction",
    page_icon=Image.open(ICON_PATH) if ICON_PATH.exists() else "❤️",
    layout="wide",
)

render_header()

if not MODEL_PATH.exists():
    st.error(
        "The trained model file svc_trained_model.pkl was not found. "
        "Run Step 6.4 of the notebook and place the file in the same folder as app.py."
    )
    st.stop()

MODEL = load_model()
ENCODERS = build_encoders()
DATASET = load_dataset()
METRICS = load_metrics()

clear_invalid_choices(DATASET)
render_inputs(DATASET)
render_predict_button(MODEL, ENCODERS, DATASET)
render_output(METRICS)
render_metrics(METRICS)

st.html('<p class="hd-footer">Heart Disease Prediction System · Educational use only</p>')
