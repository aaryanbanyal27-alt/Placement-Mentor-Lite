"""
Placement Mentor Lite - Streamlit Dashboard (Upgraded)

Improvements applied in this file:
  1. Model Comparison table  (Logistic Regression / Decision Tree / Random Forest)
  2. Feature Importance chart (from Random Forest)
  3. Placement Readiness Grade (Excellent / Good / Average / Needs Improvement)
  4. Personalized Roadmap (profile-aware gap detection in roadmap.py)
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analysis import TARGET_ROLE_DATA, analyze_student_profile
from src.ml_model import predict_placement_probability, train_placement_model
from src.storage import load_student_profiles, save_student_profile


st.set_page_config(page_title="Placement Mentor Lite", page_icon="🎯", layout="wide")

st.title("🎯 Placement Mentor Lite")
st.caption("AI-powered career readiness dashboard for 2nd-year AI & Data Science students")


@st.cache_resource
def get_model_bundle():
    """Train all three models once and cache them for the session."""
    return train_placement_model()


# ── Sidebar: Student Profile Input ──────────────────────────────────────────
with st.sidebar:
    st.header("📋 Student Profile")

    name = st.text_input("Name", value="Aaryan")
    roll_no = st.text_input("Roll No", value="AIDS042")
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.2, step=0.1)
    skills = st.text_area("Skills (comma separated)", value="Python, Excel, Basic SQL")
    projects = st.number_input("Projects completed", min_value=0, max_value=20, value=2, step=1)
    internship_status = st.selectbox(
        "Internship status",
        ["No Internship", "Applied", "Doing Internship", "Completed Internship"],
        index=2,
    )
    coding_platform_activity = st.slider(
        "Coding platform activity",
        min_value=0,
        max_value=300,
        value=75,
        help="Approximate number of solved questions or meaningful practice submissions.",
    )
    target_role = st.selectbox("Target role", list(TARGET_ROLE_DATA.keys()), index=4)
    submitted = st.button("Generate Mentor Report", type="primary", use_container_width=True)


# ── Core Analysis ────────────────────────────────────────────────────────────
profile = {
    "name": name,
    "roll_no": roll_no,
    "cgpa": cgpa,
    "skills": skills,
    "projects": int(projects),
    "internship_status": internship_status,
    "coding_platform_activity": int(coding_platform_activity),
    "target_role": target_role,
}

report = analyze_student_profile(profile)
model_bundle = get_model_bundle()
placement_probability = predict_placement_probability(model_bundle, report["saved_profile"])

if submitted:
    save_student_profile(report["saved_profile"])
    st.success("✅ Student profile saved to data/students.csv")


# ── Top Metrics ──────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Readiness Score", f"{report['readiness_score']}/100")
col2.metric("🏅 Grade", report["readiness_grade"])
col3.metric("🎯 Career Completion", f"{report['completion_percentage']}%")
col4.metric("🤖 ML Placement Probability", f"{placement_probability}%")
st.markdown("---")


# ── Section 1: Gap Analysis + Readiness Breakdown ────────────────────────────
left_col, right_col = st.columns([1.05, 1])

with left_col:
    st.subheader("🔍 Gap Analysis")
    st.write(f"Student wants: **{target_role}**")
    st.write("Skills required for this role:")
    for skill in report["required_skills"]:
        marker = "✅ Available" if skill in report["matched_skills"] else "❌ Missing"
        st.write(f"- {skill} — {marker}")

    st.progress(report["completion_percentage"] / 100)
    st.write(f"Current skill completion: **{report['completion_percentage']}%**")

    col_s, col_w = st.columns(2)
    with col_s:
        st.subheader("💪 Strengths")
        if report["strengths"]:
            for s in report["strengths"]:
                st.write(f"- {s}")
        else:
            st.info("Start building your first strengths this week.")
    with col_w:
        st.subheader("⚠️ Weaknesses")
        if report["weaknesses"]:
            for w in report["weaknesses"]:
                st.write(f"- {w}")
        else:
            st.success("No major weakness detected. Keep polishing.")

with right_col:
    st.subheader("📈 Readiness Breakdown")
    breakdown = pd.DataFrame({
        "Area": list(report["score_breakdown"].keys()),
        "Score": list(report["score_breakdown"].values()),
    })
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#287c8e", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    ax.barh(breakdown["Area"], breakdown["Score"], color=colors)
    ax.set_xlabel("Score contribution")
    ax.set_title("Placement readiness components")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)
    plt.close()


st.markdown("---")


# ── Section 2: Model Comparison + Feature Importance ─────────────────────────
st.subheader("🤖 Machine Learning")
ml_left, ml_right = st.columns(2)

with ml_left:
    # Improvement 1: Model Comparison Table
    st.markdown("**Model Comparison**")
    acc_data = pd.DataFrame({
        "Model": list(model_bundle["model_accuracies"].keys()),
        "Accuracy (%)": list(model_bundle["model_accuracies"].values()),
    }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    # Highlight best model
    def highlight_best(row):
        return ["background-color: #d4edda; font-weight: bold" if row["Accuracy (%)"] == acc_data["Accuracy (%)"].max() else "" for _ in row]

    st.dataframe(acc_data.style.apply(highlight_best, axis=1), use_container_width=True)
    st.caption(f"Best model: **{model_bundle['model_name']}** ({model_bundle['accuracy']}% accuracy)")

with ml_right:
    # Improvement 2: Feature Importance Chart
    st.markdown("**Feature Importance (Random Forest)**")
    fi = model_bundle["feature_importance"]
    fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance (%)": list(fi.values())})
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.barh(fi_df["Feature"], fi_df["Importance (%)"], color="#287c8e")
    ax2.set_xlabel("Importance (%)")
    ax2.set_title("What drives placement probability?")
    ax2.grid(axis="x", alpha=0.25)
    for i, (val, feat) in enumerate(zip(fi_df["Importance (%)"], fi_df["Feature"])):
        ax2.text(val + 0.3, i, f"{val}%", va="center", fontsize=9)
    st.pyplot(fig2)
    plt.close()


st.markdown("---")


# ── Section 3: Personalized Roadmap ──────────────────────────────────────────
st.subheader("🗺️ Personalized Roadmap")
tab_30, tab_60, tab_90 = st.tabs(["📅 30 Days", "📅 60 Days", "📅 90 Days"])
with tab_30:
    for i, item in enumerate(report["roadmaps"]["30_day"], start=1):
        st.write(f"{i}. {item}")
with tab_60:
    for i, item in enumerate(report["roadmaps"]["60_day"], start=1):
        st.write(f"{i}. {item}")
with tab_90:
    for i, item in enumerate(report["roadmaps"]["90_day"], start=1):
        st.write(f"{i}. {item}")


st.markdown("---")


# ── Section 4: Project + Resource Suggestions ─────────────────────────────────
st.subheader("💡 Project & Resource Suggestions")
proj_col, res_col = st.columns(2)
with proj_col:
    st.markdown("**Suggested Projects**")
    for p in report["suggested_projects"]:
        st.write(f"- {p}")
with res_col:
    st.markdown("**Learning Resources**")
    for r in report["learning_resources"]:
        st.write(f"- {r}")


st.markdown("---")


# ── Section 5: Saved Profiles Table ──────────────────────────────────────────
st.subheader("🗂️ Saved Student Profiles")
saved_profiles = load_student_profiles()
if saved_profiles.empty:
    st.info("No saved profiles yet. Click **Generate Mentor Report** to save one.")
else:
    st.dataframe(saved_profiles, use_container_width=True)
