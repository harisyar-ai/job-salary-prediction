import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback
import joblib  # Try joblib instead of pickle

st.set_page_config(
    page_title="Job Salary Predictor",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
    }

    .main-title {
        font-size: 3.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #7cffc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0 0.6rem;
        text-shadow: 0 4px 15px rgba(0, 212, 255, 0.45);
    }

    .subtitle {
        text-align: center;
        color: #a0d0ff;
        font-size: 1.3rem;
        margin-bottom: 2.2rem;
    }

    .card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.9rem;
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1.6rem;
        transition: all 0.32s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }

    .card:hover {
        transform: translateY(-9px);
        box-shadow: 0 18px 50px rgba(0, 212, 255, 0.28);
        border-color: rgba(0, 212, 255, 0.42);
    }

    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2.6rem 1.8rem;
        text-align: center;
        color: white;
        margin: 2.5rem 0;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.45);
        border: 1px solid rgba(255,255,255,0.18);
    }

    .result-amount {
        font-size: 3.8rem;
        font-weight: 900;
        margin: 0.7rem 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2a44 0%, #0f1e33 100%) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.15);
    }

    .sidebar-title {
        color: #00d4ff;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 1.6rem 0 1.8rem;
        text-align: center;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a2a44 0%, #0f1e33 100%);
        color: #00d4ff;
        font-weight: 700;
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 0.9rem 2.2rem;
        transition: all 0.28s ease;
    }

    .stButton > button:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.25);
        border-color: rgba(0, 212, 255, 0.6);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #1a2a44 0%, #0f1e33 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        transition: all 0.28s ease;
    }

    .stSelectbox > div > div:hover {
        transform: translateY(-6px);
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.25);
    }

    /* Radio button styling */
    .stRadio > div {
        background: linear-gradient(135deg, #1a2a44 0%, #0f1e33 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        transition: all 0.28s ease;
    }

    .stRadio > div:hover {
        transform: translateY(-6px);
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.25);
    }

    /* Slider styling */
    .stSlider {
        transition: all 0.28s ease;
    }

    .stSlider:hover {
        transform: translateY(-6px);
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #7cffc4) !important;
    }

    /* Number input styling */
    .stNumberInput button {
        display: none !important;
    }

    .stNumberInput [data-baseweb="input"] {
        background: linear-gradient(135deg, #1a2a44 0%, #0f1e33 100%) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.28s ease !important;
    }

    .stNumberInput:hover [data-baseweb="input"] {
        transform: translateY(-6px) !important;
        border-color: rgba(0, 212, 255, 0.6) !important;
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.25) !important;
    }

    .stNumberInput input {
        background: transparent !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
        border: none !important;
    }

    .stNumberInput input:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    .stNumberInput > div {
        transition: all 0.28s ease !important;
    }

    .stNumberInput > div > div {
        background: linear-gradient(135deg, #1a2a44 0%, #0f1e33 100%) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.28s ease !important;
    }

    .stNumberInput:hover > div > div {
        transform: translateY(-6px) !important;
        border-color: rgba(0, 212, 255, 0.6) !important;
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.25) !important;
    }

    /* Hide the "Press Enter to submit" text in number inputs */
    .stNumberInput > label > div[data-testid="stCaptionContainer"] {
        display: none !important;
    }

    hr {
        border-color: rgba(0, 212, 255, 0.22);
        margin: 2.8rem 0;
    }

    /* Team member card with hover effect */
    .team-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1rem;
        text-align: center;
        transition: all 0.28s ease;
        box-shadow: 0 6px 20px rgba(0,0,0,0.30);
    }

    .team-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.25);
        border-color: rgba(0, 212, 255, 0.45);
    }

    .team-card h2 {
        color: #00d4ff;
        font-size: 1.5rem;
        margin: 0 0 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Model loading with better diagnostics ──────────────────────────────────
@st.cache_resource
def load_model():
    path = r"job-salary-regression/model/best_salary_model.pkl"
    
    try:
        # Try joblib first (recommended for scikit-learn models)
        model = joblib.load(path)
        # st.success(" Model loaded successfully with joblib!")
        return model
    except Exception as e1:
        try:
            # Fallback to pickle
            with open(path, 'rb') as f:
                model = pickle.load(f)
            # st.success(" Model loaded successfully with pickle!")
            return model
        except FileNotFoundError:
            st.error(f" Model file not found at:\n{path}")
            return None
        except Exception as e2:
            st.error(f" Model loading failed with both joblib and pickle")
            st.error(f"Joblib error: {str(e1)}")
            st.error(f"Pickle error: {str(e2)}")
            with st.expander("Full traceback"):
                st.code(f"JOBLIB ERROR:\n{traceback.format_exc()}\n\nPICKLE ERROR:\n{str(e2)}")
            return None

model = load_model()

# Early exit if model cannot be loaded
if model is None:
    st.warning(" The prediction model could not be loaded. Prediction features are disabled.")

# ─── Prediction logic ───────────────────────────────────────────────────────
def predict_salary(input_dict):
    if model is None:
        return None

    try:
        # Create DataFrame with proper column order matching training
        df = pd.DataFrame([input_dict])
        # The pipeline handles preprocessing, so we just pass the raw features
        prediction = model.predict(df)[0]
        return float(prediction)
    except Exception as e:
        st.error(f" Prediction failed: {str(e)}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
            st.write("Input data:", input_dict)
            st.write("DataFrame shape:", df.shape)
            st.write("DataFrame columns:", df.columns.tolist())
        return None

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Job Salary Smart AI Predictor</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Make Prediction", "Recent Predictions", "Project Info", "About Us"],
        label_visibility="collapsed"
    )

# Session state
if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = []

# ─── Pages ──────────────────────────────────────────────────────────────────
if page == "Make Prediction":
    st.markdown('<h1 class="main-title">💸 Job Salary Predictor </h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered salary estimation based on your profile</p>', unsafe_allow_html=True)

    if model is None:
        st.warning(" Prediction is currently unavailable — model could not be loaded.")
    else:
        with st.form("salary_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
                experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)
                education = st.selectbox(
                    "Education Level", 
                    ["High School", "Bachelor", "Master", "PhD"]
                )

            with col2:
                job_title = st.selectbox(
                    "Job Title",
                    ["Analyst", "Engineer", "Manager", "Director"]
                )
                location = st.selectbox(
                    "Location",
                    ["Suburban", "Rural", "Urban"]
                )
                gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

            submitted = st.form_submit_button("Calculate Estimated Salary", use_container_width=True)

        if submitted:
            input_data = {
                "Age": age,
                "Experience": experience,
                "Education": education,
                "Job_Title": job_title,
                "Location": location,
                "Gender": gender
            }

            with st.spinner("Calculating..."):
                salary = predict_salary(input_data)

            if salary is not None:
                st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size:1.45rem; opacity:0.92;">Estimated Annual Salary</div>
                        <div class="result-amount">${salary:,.0f}</div>
                        <div style="font-size:1.1rem; margin-top:0.9rem; opacity:0.84;">
                            (AI model estimate — actual salaries may vary)
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.session_state.recent_predictions.append({
                    "Age": age,
                    "Experience": experience,
                    "Education": education,
                    "Job Title": job_title,
                    "Location": location,
                    "Gender": gender,
                    "Estimated Salary": f"${salary:,.0f}"
                })

elif page == "Recent Predictions":
    st.markdown('<h1 class="main-title">💸 Job Salary Predictor </h1>', unsafe_allow_html=True)

    if st.session_state.recent_predictions:
        df = pd.DataFrame(st.session_state.recent_predictions)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No predictions have been made yet.")

elif page == "Project Info":
    st.markdown('<h1 class="main-title">💸 Job Salary Predictor </h1>', unsafe_allow_html=True)

    st.markdown("""
**Job Salary Predictor** is a machine learning application that estimates 
annual salaries based on professional and demographic features.

**Main characteristics:**
- Trained on comprehensive salary data across multiple industries
- Uses advanced preprocessing with StandardScaler, OrdinalEncoder, and OneHotEncoder
- Multiple models evaluated (Linear, Ridge, Lasso, KNN, SVR, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost)
- Best performing model selected based on RMSE
- Typical test R² score: High accuracy with cross-validation
- Modern interface built with Streamlit

**Features considered:**
- Age and Years of Experience
- Education Level (High School to PhD)
- Job Title (Analyst, Engineer, Manager, Director)
- Geographic Location (Suburban, Rural, Urban)
- Gender demographics

**Important:** This is only an estimate. Real salaries depend on many factors including company size, 
industry, specific skills, negotiation, and market conditions not included in this model.
    """)

elif page == "About Us":
    st.markdown('<h1 class="main-title">💸 Job Salary Predictor </h1>', unsafe_allow_html=True)

    cols = st.columns(2)

    with cols[0]:
        st.markdown('''
        <div class="team-card">
            <h2>Haider Haroon</h2>
            <p style="text-align: left; margin: 0;">
                • AI Engineer<br>
                • Specializing in AI Security & Computer Vision<br>
                • Passionate about building secure, real-world AI systems<br>
                • Experienced in deploying AI solutions across various industries
            </p>
        </div>
        ''', unsafe_allow_html=True)

    with cols[1]:
        st.markdown('''
        <div class="team-card">
            <h2>Haris Yar</h2>
            <p style="text-align: left; margin: 0;">
                • Self-taught AI Developer<br>
                • Enthusiast in Machine Learning and Computer Vision<br>
                • Committed to continuous learning and innovation in AI Research<br>
                • Aspiring to contribute to impactful AI solutions
            </p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Connect")
    st.markdown("""
**Haider Haroon**  
GitHub: https://github.com/Haid3rH    
Email: haiderharoon2005@gmail.com
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#a0d0ff; padding:2rem 0; font-size:0.95rem;">
        © Built by Haider • Haris | 2026
    </div>
""", unsafe_allow_html=True)