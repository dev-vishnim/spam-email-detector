"""Spam Email Detector - Streamlit UI for the Random Forest (15 features) model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from feature_extractor import extract_features

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "random_forest_spam_final_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "selected_features.pkl"
RESULTS_DIR = BASE_DIR / "results"

SAMPLE_SPAM = (
    "CONGRATULATIONS!!! You have WON $1,000,000 in our FREE prize draw!!! "
    "Click here NOW to receive your money. Your account will be credited today. "
    "Reply with your details or click to remove your name from the list. ACT NOW!!! $$$"
)
SAMPLE_HAM = (
    "Hi team, the project review is on Thursday at 2pm in the lab. Please update your "
    "tasks on the board before then and bring the notes we discussed last week. "
    "Thanks, Sam"
)

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

st.markdown(
    """
    <style>
    .verdict {padding: 1.2rem 1.4rem; border-radius: 14px; margin-bottom: .5rem;}
    .verdict h2 {margin: 0 0 .2rem 0; font-size: 1.7rem;}
    .verdict p {margin: 0; opacity: .85;}
    .spam {background:#fdecec; border:1px solid #f3b4b4; color:#8a1c1c;}
    .ham  {background:#e8f7ee; border:1px solid #a9dcbc; color:#14532d;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


model, FEATURES = load_artifacts()
importance = pd.Series(model.feature_importances_, index=FEATURES)


def predict(values: dict[str, float]) -> float:
    """Return the probability that the email is spam."""
    row = pd.DataFrame([values])[FEATURES]
    return float(model.predict_proba(row)[0, 1])


def show_verdict(prob: float, threshold: float) -> None:
    is_spam = prob >= threshold
    css, label, icon = ("spam", "SPAM", "🚨") if is_spam else ("ham", "NOT SPAM", "✅")
    st.markdown(
        f'<div class="verdict {css}"><h2>{icon} {label}</h2>'
        f"<p>Spam probability: <b>{prob:.1%}</b> (decision threshold {threshold:.0%})</p></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(prob, 0.0), 1.0))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Spam threshold",
        0.05, 0.95, 0.50, 0.05,
        help="An email is flagged as spam when its spam probability is at or above this value. "
             "Lower = catches more spam but flags more good emails (higher recall, lower precision). "
             "Higher = the opposite.",
    )
    st.divider()
    st.caption("**Model:** Random Forest, 200 trees")
    st.caption(f"**Inputs:** {len(FEATURES)} of 57 Spambase features")
    st.caption("**Training:** SMOTE on the training split, tuned with GridSearchCV")

# ---------------------------------------------------------------- header
st.title("📧 Spam Email Detector")
st.write("Paste an email and the Random Forest model will classify it as **spam** or **not spam**.")

tab_email, tab_manual, tab_model = st.tabs(["✉️ Check an email", "🎛️ Enter features manually", "📊 About the model"])

# ---------------------------------------------------------------- tab 1: raw text
with tab_email:
    if "email_text" not in st.session_state:
        st.session_state.email_text = ""

    def _set_text(value: str) -> None:
        st.session_state.email_text = value

    c1, c2, c3 = st.columns([1, 1, 4])
    c1.button("Load spam example", on_click=_set_text, args=(SAMPLE_SPAM,))
    c2.button("Load normal example", on_click=_set_text, args=(SAMPLE_HAM,))

    st.text_area(
        "Email content (subject + body)",
        key="email_text",
        height=220,
        placeholder="Paste the full text of an email here...",
    )

    text = st.session_state.email_text
    if st.button("Analyze email", type="primary", disabled=not text.strip()):
        values = extract_features(text, FEATURES)
        prob = predict(values)

        left, right = st.columns([1, 1.4], gap="large")
        with left:
            show_verdict(prob, threshold)
            st.caption(
                "⚠️ The model was trained on the Spambase dataset (older emails collected at HP Labs). "
                "Words such as *george*, *hp*, *hpl* and *edu* push the score toward 'not spam' "
                "because of where that data came from."
            )
        with right:
            st.subheader("What the model saw")
            detail = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Value in this email": [values[f] for f in FEATURES],
                    "Importance in model": [importance[f] for f in FEATURES],
                }
            ).sort_values("Importance in model", ascending=False)
            st.dataframe(
                detail,
                hide_index=True,
                width="stretch",
                column_config={
                    "Value in this email": st.column_config.NumberColumn(format="%.2f"),
                    "Importance in model": st.column_config.ProgressColumn(
                        format="%.3f", min_value=0.0, max_value=float(importance.max())
                    ),
                },
            )
            st.caption(
                "Word and character values are percentages (e.g. `word_freq_free = 5.0` means 5% of the "
                "words are 'free'). Capital-run features describe runs of consecutive UPPERCASE letters."
            )

# ---------------------------------------------------------------- tab 2: manual features
with tab_manual:
    st.write("Set the 15 feature values yourself, exactly as the model receives them.")
    manual: dict[str, float] = {}
    cols = st.columns(3)
    for i, name in enumerate(FEATURES):
        is_capital = name.startswith("capital_run_length")
        manual[name] = cols[i % 3].number_input(
            name,
            min_value=1.0 if is_capital else 0.0,
            value=1.0 if is_capital else 0.0,
            step=1.0 if is_capital else 0.1,
            key=f"manual_{name}",
        )
    if st.button("Predict from features", type="primary"):
        show_verdict(predict(manual), threshold)

# ---------------------------------------------------------------- tab 3: model info
with tab_model:
    st.subheader("Test-set performance")
    comparison = load_csv("random_forest_model_comparison.csv")
    st.dataframe(
        comparison.style.format({c: "{:.4f}" for c in comparison.columns if c != "Model"}),
        hide_index=True, width="stretch",
    )
    st.caption(
        "The deployed model is the **15-feature tuned model**. It uses 73.7% fewer inputs than the "
        "57-feature baseline for a small drop in accuracy."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Feature importance (deployed model)")
        st.bar_chart(importance.sort_values(), horizontal=True)
    with right:
        st.subheader("Cross-validated F1 by feature count")
        safe = load_csv("random_forest_leakage_safe_validation.csv").set_index("Features")["Mean CV F1"]
        st.line_chart(safe.rename("Leakage-safe CV F1"))
        st.caption("SMOTE is applied inside each fold, so these estimates are not inflated by leakage.")

    with st.expander("Limitations"):
        st.markdown(
            "- Spambase stores numeric features, not raw text. This app recreates those features from pasted text, "
            "so results are close to, but not guaranteed identical to, the original dataset's extraction.\n"
            "- The training emails are old and come from one source, so modern spam may be missed.\n"
            "- SMOTE creates synthetic samples that may not perfectly resemble real spam.\n"
            "- Use this as a coursework demo, not as a production spam filter."
        )
