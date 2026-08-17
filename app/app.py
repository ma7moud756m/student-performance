from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "student_performance_model.pkl"


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #071018;
            --panel: rgba(15, 28, 42, 0.78);
            --panel-strong: rgba(18, 35, 52, 0.92);
            --line: rgba(125, 232, 236, 0.16);
            --text: #edf7fb;
            --muted: #94a8b8;
            --accent: #4de1df;
            --accent-2: #72a7ff;
            --good: #75e0a7;
            --warn: #ffd166;
            --risk: #ff7a90;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 8%, rgba(77, 225, 223, 0.15), transparent 28%),
                radial-gradient(circle at 84% 18%, rgba(114, 167, 255, 0.13), transparent 26%),
                linear-gradient(135deg, #071018 0%, #0a1521 48%, #09111a 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: rgba(5, 13, 22, 0.94);
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] * {
            color: var(--text);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(12, 26, 40, 0.72);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.35rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(260px, 0.55fr);
            gap: 1.6rem;
            align-items: center;
            padding: 2.1rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(125, 232, 236, 0.18);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(18, 35, 52, 0.92), rgba(10, 22, 34, 0.72));
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.25rem, 5vw, 4.4rem);
            line-height: 1;
            font-weight: 800;
            color: var(--text);
        }

        .hero p {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            margin: 0.8rem 0 0;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            width: fit-content;
            padding: 0.42rem 0.72rem;
            margin-bottom: 0.95rem;
            border: 1px solid rgba(77, 225, 223, 0.28);
            border-radius: 999px;
            background: rgba(77, 225, 223, 0.08);
            color: #c9fbff;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .orbital {
            min-height: 230px;
            position: relative;
            border-radius: 8px;
            background:
                linear-gradient(145deg, rgba(77, 225, 223, 0.12), rgba(114, 167, 255, 0.07)),
                rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(125, 232, 236, 0.16);
        }

        .orbital:before,
        .orbital:after {
            content: "";
            position: absolute;
            inset: 28px;
            border: 1px solid rgba(77, 225, 223, 0.22);
            border-radius: 50%;
            transform: rotate(-18deg);
        }

        .orbital:after {
            inset: 58px;
            border-color: rgba(114, 167, 255, 0.26);
            transform: rotate(24deg);
        }

        .node {
            position: absolute;
            width: 54px;
            height: 54px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: #041018;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            box-shadow: 0 0 34px rgba(77, 225, 223, 0.25);
        }

        .node.one { top: 24px; right: 58px; }
        .node.two { left: 38px; bottom: 36px; }
        .node.three { right: 76px; bottom: 28px; width: 40px; height: 40px; font-size: 0.85rem; }

        .section-card {
            padding: 1.25rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
        }

        .section-card h2,
        .section-card h3 {
            margin-top: 0;
        }

        .muted {
            color: var(--muted);
        }

        .prediction-card {
            padding: 1.35rem;
            border: 1px solid rgba(77, 225, 223, 0.24);
            border-radius: 8px;
            background:
                linear-gradient(160deg, rgba(77, 225, 223, 0.13), rgba(114, 167, 255, 0.08)),
                var(--panel-strong);
            box-shadow: 0 22px 58px rgba(0, 0, 0, 0.28);
        }

        .score {
            display: flex;
            align-items: baseline;
            gap: 0.55rem;
            margin: 0.5rem 0 0.2rem;
        }

        .score .value {
            font-size: clamp(3rem, 7vw, 5.4rem);
            font-weight: 800;
            line-height: 1;
            color: var(--text);
        }

        .score .scale {
            color: var(--muted);
            font-size: 1.25rem;
            font-weight: 700;
        }

        .category {
            display: inline-flex;
            padding: 0.42rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-weight: 700;
            margin: 0.5rem 0 0.9rem;
        }

        .category.excellent { color: var(--good); background: rgba(117, 224, 167, 0.1); }
        .category.good { color: var(--accent); background: rgba(77, 225, 223, 0.1); }
        .category.average { color: var(--warn); background: rgba(255, 209, 102, 0.1); }
        .category.needs { color: var(--risk); background: rgba(255, 122, 144, 0.1); }

        .progress-shell {
            height: 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            box-shadow: 0 0 24px rgba(77, 225, 223, 0.35);
        }

        .insight-card {
            min-height: 150px;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--line);
            background: rgba(12, 26, 40, 0.65);
        }

        .insight-card .icon {
            font-size: 1.35rem;
            margin-bottom: 0.6rem;
        }

        .input-section {
            padding-top: 0.35rem;
            margin-top: 0.55rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .input-section h4 {
            margin: 0.7rem 0 0.4rem;
            color: #d9f8fb;
            font-size: 0.9rem;
        }

        .stButton > button {
            width: 100%;
            border: 0;
            border-radius: 8px;
            padding: 0.78rem 1rem;
            color: #031015;
            font-weight: 800;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            box-shadow: 0 12px 30px rgba(77, 225, 223, 0.22);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 38px rgba(77, 225, 223, 0.28);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(12, 26, 40, 0.58);
        }

        @media (max-width: 900px) {
            .hero {
                grid-template-columns: 1fr;
                padding: 1.35rem;
            }

            .orbital {
                min-height: 160px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_level(value: float, low: float, high: float) -> str:
    if value < low:
        return "Low"
    if value < high:
        return "Medium"
    return "High"


def get_score_category(score: float) -> tuple[str, str]:
    if score < 60:
        return "Needs Improvement", "needs"
    if score < 70:
        return "Average Performance", "average"
    if score < 80:
        return "Good Performance", "good"
    return "Excellent Performance", "excellent"


def create_input_features(values: dict) -> pd.DataFrame:
    features = {
        "Hours_Studied": values["hours_studied"],
        "Attendance": values["attendance"],
        "Parental_Involvement": values["parental_involvement"],
        "Access_to_Resources": values["access_to_resources"],
        "Extracurricular_Activities": values["extracurricular_activities"],
        "Sleep_Hours": values["sleep_hours"],
        "Previous_Scores": values["previous_scores"],
        "Motivation_Level": values["motivation_level"],
        "Internet_Access": values["internet_access"],
        "Tutoring_Sessions": values["tutoring_sessions"],
        "Family_Income": values["family_income"],
        "Teacher_Quality": values["teacher_quality"],
        "School_Type": values["school_type"],
        "Peer_Influence": values["peer_influence"],
        "Physical_Activity": values["physical_activity"],
        "Learning_Disabilities": values["learning_disabilities"],
        "Parental_Education_Level": values["parental_education_level"],
        "Distance_from_Home": values["distance_from_home"],
        "Gender": values["gender"],
    }
    return pd.DataFrame([features])


def predict_score(model, features: pd.DataFrame) -> float:
    if hasattr(model, "feature_names_in_"):
        features = features.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = float(model.predict(features)[0])
    return float(np.clip(prediction, 0, 100))


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Student Performance Predictor")
        st.caption("AI-powered regression dashboard")
        page = st.radio(
            "Navigation",
            ["Prediction", "Model Insights", "About the Project"],
            label_visibility="collapsed",
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("Built with Python • Scikit-learn • Streamlit")
    return page


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="badge">AI • Regression • Machine Learning</div>
                <h1>Student Performance Predictor</h1>
                <p><strong>Predict academic performance using data-driven insights.</strong></p>
                <p>Enter the student's academic, behavioral, and environmental information to estimate their expected exam score.</p>
            </div>
            <div class="orbital" aria-hidden="true">
                <div class="node one">AI</div>
                <div class="node two">Σ</div>
                <div class="node three">R²</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inputs() -> dict:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## Student Profile")

    st.markdown('<div class="input-section"><h4>Academic Performance</h4></div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        hours_studied = st.slider("Hours Studied", 0, 45, 23)
        previous_scores = st.slider("Previous Scores", 0, 100, 73)
    with a2:
        attendance = st.slider("Attendance", 0, 100, 84, help="Attendance percentage")
        tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1, step=1)

    st.markdown('<div class="input-section"><h4>Lifestyle</h4></div>', unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        sleep_hours = st.slider("Sleep Hours", 3, 12, 7)
    with l2:
        physical_activity = st.slider("Physical Activity", 0, 6, 3, help="Sessions per week")

    st.markdown('<div class="input-section"><h4>Academic Environment</h4></div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
        parental_education_level = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"],
            index=1,
        )
    with e2:
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
        school_type = st.radio("School Type", ["Public", "Private"], horizontal=True)
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=0)
    with e3:
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=2)

    st.markdown('<div class="input-section"><h4>Personal Factors</h4></div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
    with p2:
        extracurricular_activities = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)
        internet_access = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
    with p3:
        learning_disabilities = st.radio("Learning Disabilities", ["No", "Yes"], horizontal=True)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

    values = {
        "hours_studied": hours_studied,
        "attendance": attendance,
        "previous_scores": previous_scores,
        "tutoring_sessions": tutoring_sessions,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "parental_involvement": parental_involvement,
        "access_to_resources": access_to_resources,
        "teacher_quality": teacher_quality,
        "school_type": school_type,
        "peer_influence": peer_influence,
        "parental_education_level": parental_education_level,
        "distance_from_home": distance_from_home,
        "motivation_level": motivation_level,
        "extracurricular_activities": extracurricular_activities,
        "internet_access": internet_access,
        "learning_disabilities": learning_disabilities,
        "gender": gender,
        "family_income": family_income,
    }
    values["study_level"] = get_level(hours_studied, 15, 25)
    values["attendance_level"] = get_level(attendance, 70, 85)
    return values


def render_metrics() -> None:
    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", "0.77")
    m2.metric("RMSE", "1.80")
    m3.metric("MAE", "0.45")
    m4.metric("Model", "Linear Regression")


def render_prediction_card(score: float | None, category: str = "Waiting for input", style: str = "good") -> None:
    display_score = "—" if score is None else f"{score:.1f}"
    progress = 0 if score is None else int(round(score))

    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown("## Predicted Exam Score")
    st.markdown(
        f"""
        <div class="score">
            <span class="value">{display_score}</span>
            <span class="scale">/ 100</span>
        </div>
        <div class="category {style}">{category}</div>
        <div class="progress-shell">
            <div class="progress-fill" style="width: {progress}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    render_metrics()
    st.markdown("</div>", unsafe_allow_html=True)


def render_insights() -> None:
    st.markdown("## What Influences Performance?")
    insights = [
        ("01", "Attendance has the strongest linear relationship with Exam Score."),
        ("02", "Hours Studied is the second strongest numerical factor."),
        ("03", "Previous Scores and Tutoring Sessions show weaker positive relationships."),
        ("04", "Categorical factors generally show smaller differences in average Exam Score."),
    ]
    cols = st.columns(4)
    for col, (icon, text) in zip(cols, insights):
        with col:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="icon">{icon}</div>
                    <div>{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_exploration(values: dict, score: float | None, category: str) -> None:
    with st.expander("Explore Prediction", expanded=False):
        summary = pd.DataFrame(
            {
                "Feature": [
                    "Hours Studied",
                    "Attendance",
                    "Study Level",
                    "Attendance Level",
                    "Previous Scores",
                    "Tutoring Sessions",
                    "Sleep Hours",
                    "Physical Activity",
                    "Motivation Level",
                    "Access to Resources",
                ],
                "Value": [
                    values["hours_studied"],
                    f'{values["attendance"]}%',
                    values["study_level"],
                    values["attendance_level"],
                    values["previous_scores"],
                    values["tutoring_sessions"],
                    values["sleep_hours"],
                    values["physical_activity"],
                    values["motivation_level"],
                    values["access_to_resources"],
                ],
            }
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with c2:
            st.metric("Predicted Score", "—" if score is None else f"{score:.1f} / 100")
            st.metric("Score Category", category)
            st.progress(0 if score is None else min(max(score / 100, 0), 1))


def render_about() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## About the Project")
    st.markdown(
        """
        This project uses machine learning regression techniques to predict student exam
        performance from academic, behavioral, and environmental factors.

        Multiple regression models were evaluated, and Linear Regression achieved the best
        overall performance on the test set. The application loads the saved Joblib model
        directly and does not retrain inside the Streamlit interface.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()
    page = render_sidebar()
    render_hero()

    if page == "Prediction":
        left, right = st.columns([1.35, 0.9], gap="large")
        with left:
            values = render_inputs()

        model = load_model()
        with right:
            score = st.session_state.get("predicted_score")
            category, style = get_score_category(score) if score is not None else ("—", "good")

            if st.button("Predict Exam Score", type="primary"):
                if model is None:
                    st.error("Model file not found. Add student_performance_model.pkl to the app folder.")
                else:
                    features = create_input_features(values)
                    score = predict_score(model, features)
                    st.session_state["predicted_score"] = score
                    category, style = get_score_category(score)

            render_prediction_card(score, category, style)

        st.markdown("<br>", unsafe_allow_html=True)
        render_insights()
        render_exploration(values, score, category)

    elif page == "Model Insights":
        render_insights()
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Model Summary")
        render_metrics()
        st.markdown(
            '<p class="muted">The dashboard emphasizes interpretable regression performance and feature behavior from the project analysis.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        render_about()


if __name__ == "__main__":
    main()
