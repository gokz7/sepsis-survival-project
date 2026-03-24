# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st                                 # for building the web app
import pandas as pd                                    # for data manipulation
import numpy as np                                     # for numerical operations
import joblib                                          # for loading saved model and scaler
import shap                                            # for explainable ai charts
import matplotlib                                      # for chart backend
matplotlib.use('Agg')                                  # non interactive backend for streamlit
import matplotlib.pyplot as plt                        # for creating charts
from datetime import datetime                          # for report timestamp
import warnings                                        # suppress warnings
warnings.filterwarnings('ignore')                      # hide warnings

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sepsis Survival Platform",             # browser tab title
    page_icon="🏥",                                    # browser tab icon
    layout="wide"                                      # wide layout for better display
)

# ── Load Model and Scaler ─────────────────────────────────────────────────────

@st.cache_resource                                     # caching so model loads only once
def load_assets():
    import os
    # checking if running from app folder or project root
    if os.path.exists('models/best_model.pkl'):
        model_path  = 'models/best_model.pkl'
        scaler_path = 'models/scaler.pkl'
    else:
        model_path  = '../models/best_model.pkl'
        scaler_path = '../models/scaler.pkl'

    model  = joblib.load(model_path)                   # loading best trained model
    scaler = joblib.load(scaler_path)                  # loading fitted scaler
    return model, scaler

model, scaler = load_assets()

# ── Feature Engineering Function ─────────────────────────────────────────────
# must match exactly what was done during training

def engineer_features(df):
    df = df.copy()                                     # copy to avoid modifying original

    df['age_group'] = pd.cut(df['age_years'],
                              bins=[0,18,40,60,80,100],
                              labels=[0,1,2,3,4],
                              include_lowest=True).astype(int)

    df['is_elderly']        = (df['age_years'] >= 65).astype(int)
    df['is_child']          = (df['age_years'] <= 18).astype(int)
    df['is_very_elderly']   = (df['age_years'] >= 80).astype(int)
    df['age_episode_ratio'] = df['age_years'] / df['episode_number']
    df['age_x_episode']     = df['age_years'] * df['episode_number']
    df['elderly_x_episode'] = df['is_elderly'] * df['episode_number']
    df['age_squared']       = df['age_years'] ** 2

    return df

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏥 Sepsis Survival & Risk Prediction Platform")
st.markdown("**AI-powered Clinical Decision Support System for ICU Teams**")
st.markdown("---")

# ── Sidebar Patient Input Form ────────────────────────────────────────────────

with st.sidebar:
    st.header("📋 Patient Input Form")
    st.markdown("Enter patient clinical details below:")
    st.markdown("---")

    age_years = st.number_input(
        "Age (Years)",
        min_value=0,
        max_value=120,
        value=50,                                      # default value
        help="Enter patient age in years. Age 0 = infant under 1 year")

    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"],
        help="Patient biological sex")
    sex_0male_1female = 0 if sex == "Male" else 1      # encoding to match training data

    episode_number = st.number_input(
        "Episode Number",
        min_value=1,
        max_value=5,
        value=1,
        help="How many times has patient been admitted for sepsis")

    st.markdown("---")
    predict_button = st.button(
        "🔍 Assess Patient Risk",
        type="primary",
        use_container_width=True)                      # button stretches full sidebar width

# ── Default Landing Page ──────────────────────────────────────────────────────

if not predict_button:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Step 1**\n\nEnter patient age, sex and episode number in the sidebar")
    with col2:
        st.info("**Step 2**\n\nClick the Assess Patient Risk button")
    with col3:
        st.info("**Step 3**\n\nReview AI prediction, risk level and SHAP explanation")

    st.markdown("---")
    st.subheader("ℹ️ About This Platform")
    st.markdown("""
    This platform uses machine learning to predict sepsis patient survival probability.

    | Feature | Detail |
    |---------|--------|
    | Dataset | 110,204 real hospital records |
    | Model | Random Forest with SMOTE balancing |
    | Validation | Tested on 3 separate cohorts |
    | Explainability | SHAP values for every prediction |

    ⚠️ **Disclaimer:** This tool supports clinical decisions.
    Final decisions must always be made by qualified medical professionals.
    """)

# ── Prediction Logic ──────────────────────────────────────────────────────────

if predict_button:

    # build raw input dataframe
    raw_df = pd.DataFrame({
        'age_years'         : [age_years],
        'sex_0male_1female' : [sex_0male_1female],
        'episode_number'    : [episode_number]
    })

    # apply feature engineering
    engineered_df = engineer_features(raw_df)

    # align columns to match scaler
    if hasattr(scaler, 'feature_names_in_'):
        engineered_df = engineered_df[scaler.feature_names_in_]  # reorder columns to match scaler

    # scale the input
    scaled_arr = scaler.transform(engineered_df)       # scaling using fitted scaler
    scaled_df  = pd.DataFrame(scaled_arr,
                               columns=engineered_df.columns)  # keeping column names for shap

    # get prediction
    death_prob    = model.predict_proba(scaled_df)[0][0]   # index 0 = dead class probability
    survival_prob = 1 - death_prob                         # survival probability

    # risk category
    if death_prob >= 0.60:
        risk_category = "🔴 HIGH RISK"
        risk_color    = "red"
        icu_advice    = "Immediate ICU admission required. Alert senior physician urgently. Begin aggressive sepsis protocol."
    elif death_prob >= 0.30:
        risk_category = "🟡 MEDIUM RISK"
        risk_color    = "orange"
        icu_advice    = "Close monitoring required. Consider ICU transfer. Reassess vitals every 2 hours."
    else:
        risk_category = "🟢 LOW RISK"
        risk_color    = "green"
        icu_advice    = "Continue standard monitoring. Reassess in 4 hours. Maintain current treatment plan."

    # ── Section 1: Key Metrics ────────────────────────────────────────────────

    st.subheader("📊 Assessment Results")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Death Probability",
            value=f"{death_prob*100:.1f}%")            # showing death probability

    with col2:
        st.metric(
            label="Survival Probability",
            value=f"{survival_prob*100:.1f}%")         # showing survival probability

    with col3:
        st.metric(
            label="Patient Age",
            value=f"{age_years} yrs")                  # patient age

    with col4:
        st.metric(
            label="Episode Number",
            value=f"{episode_number}")                 # episode number

    st.markdown("---")

    # ── Section 2: Risk Category and ICU Advice ───────────────────────────────

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("🎯 Risk Category")
        if risk_color == "red":
            st.error(f"**{risk_category}**")           # red box for high risk
        elif risk_color == "orange":
            st.warning(f"**{risk_category}**")         # orange box for medium risk
        else:
            st.success(f"**{risk_category}**")         # green box for low risk

        st.progress(float(death_prob))                 # progress bar showing death probability

    with col6:
        st.subheader("🏥 ICU Recommendation")
        if risk_color == "red":
            st.error(f"⚠️ {icu_advice}")               # red recommendation for high risk
        elif risk_color == "orange":
            st.warning(f"⚠️ {icu_advice}")             # orange recommendation
        else:
            st.success(f"✅ {icu_advice}")              # green recommendation

    st.markdown("---")

    # ── Section 3: Probability Chart ─────────────────────────────────────────

    col7, col8 = st.columns(2)

    with col7:
        st.subheader("📈 Survival Probability Chart")
        fig1, ax1 = plt.subplots(figsize=(6, 3))

        bars = ax1.barh(
            ['Death Risk', 'Survival'],
            [death_prob, survival_prob],
            color=['#e74c3c', '#2ecc71'],               # red for death, green for survival
            height=0.4)

        ax1.set_xlim(0, 1)                             # x axis from 0 to 1
        ax1.axvline(x=0.5, color='black',
                    linestyle='--', alpha=0.5,
                    label='50% threshold')             # threshold line

        for bar, val in zip(bars, [death_prob, survival_prob]):
            ax1.text(val + 0.01,
                     bar.get_y() + bar.get_height()/2,
                     f'{val*100:.1f}%',
                     va='center', fontsize=12,
                     fontweight='bold')

        ax1.set_xlabel('Probability')
        ax1.set_title('Death vs Survival Probability')
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col8:
        st.subheader("📋 Patient Risk Summary")

        # determining age category for display
        if age_years <= 18:
            age_cat = "Child"
        elif age_years <= 40:
            age_cat = "Young Adult"
        elif age_years <= 60:
            age_cat = "Middle Aged"
        elif age_years <= 80:
            age_cat = "Senior"
        else:
            age_cat = "Very Elderly"

        summary_df = pd.DataFrame({
            'Parameter' : ['Age', 'Age Category', 'Sex',
                          'Episode Number', 'Elderly Flag',
                          'Very Elderly Flag'],
            'Value'     : [
                f"{age_years} years",
                age_cat,
                sex,
                episode_number,
                "Yes" if age_years >= 65 else "No",
                "Yes" if age_years >= 80 else "No"
            ]
        })
        st.dataframe(summary_df,
                     use_container_width=True,
                     hide_index=True)                  # hiding row index

    st.markdown("---")

    # ── Section 4: SHAP Explanation ───────────────────────────────────────────

    st.subheader("🧠 AI Explanation — Why This Prediction?")
    st.caption("SHAP values show which features contributed most to this prediction")

    try:
        explainer  = shap.TreeExplainer(model)         # tree explainer works best for random forest
        shap_vals  = explainer.shap_values(scaled_df)  # calculating shap values

        # for binary classification random forest returns list of 2 arrays
        if isinstance(shap_vals, list):
            sv = shap_vals[0][0]                       # index 0 = dead class shap values
        else:
            sv = shap_vals[0]

        feature_names = engineered_df.columns.tolist() # getting feature names

        # sorting by absolute shap value for better visualization
        sorted_idx   = np.argsort(np.abs(sv))          # sort by magnitude
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_vals  = [sv[i] for i in sorted_idx]

        colors = ['#e74c3c' if v > 0 else '#2ecc71'
                  for v in sorted_vals]                # red increases risk, green decreases

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(sorted_names, sorted_vals,
                 color=colors, alpha=0.8)
        ax2.axvline(x=0, color='black', linewidth=0.8) # zero line
        ax2.set_xlabel('SHAP Value\n(Positive = increases death risk | Negative = decreases death risk)')
        ax2.set_title('Feature Contribution to Prediction')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.caption("🔴 Red bars increase death risk | 🟢 Green bars decrease death risk")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    st.markdown("---")

    # ── Section 5: Download Report ────────────────────────────────────────────

    st.subheader("📥 Download Prediction Report")

    report = f"""
SEPSIS SURVIVAL & RISK PREDICTION REPORT
==========================================
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PATIENT DETAILS
---------------
Age            : {age_years} years ({age_cat})
Sex            : {sex}
Episode Number : {episode_number}
Elderly Flag   : {"Yes" if age_years >= 65 else "No"}
Very Elderly   : {"Yes" if age_years >= 80 else "No"}

PREDICTION RESULTS
------------------
Death Probability    : {death_prob*100:.2f}%
Survival Probability : {survival_prob*100:.2f}%
Risk Category        : {risk_category}

ICU RECOMMENDATION
------------------
{icu_advice}

MODEL INFORMATION
-----------------
Model Used     : Random Forest Classifier
Training Data  : 110,204 hospital records
Validation AUC : 0.6076
Techniques     : SMOTE + Stratified K-Fold + Class Weighting

DISCLAIMER
----------
This report is generated by an AI decision support tool.
All clinical decisions must be made by qualified medical professionals.
This tool does not replace physician judgment.
==========================================
    """

    st.download_button(
        label="📥 Download Full Report (TXT)",
        data=report,
        file_name=f"sepsis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True)