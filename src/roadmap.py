"""
src/roadmap.py
==============
Personalized 30/60/90-day roadmap generator — upgraded with type hints.

Design philosophy:
    A generic roadmap ("learn Python, do projects") is useless because it
    ignores where the student already is.  This module generates a roadmap
    that is different for every student by detecting their specific gaps
    across five dimensions: skills, CGPA, projects, coding activity,
    and internship status.

Public API:
    generate_roadmaps(role, missing_skills, projects, profile) → dict
"""

from typing import Any, Dict, List, Optional


# ── Skill-specific task library ───────────────────────────────────────────────
# Maps each required role skill to a concrete, actionable 30-day task.
# If a student is missing that skill, the corresponding task is added to their plan.

_SKILL_TASKS: dict[str, str] = {
    "SQL":
        "Learn SQL basics: SELECT, JOINs, GROUP BY — practice on SQLBolt",
    "Pandas":
        "Learn Pandas: DataFrames, groupby, merge — Kaggle micro-course",
    "Data Visualization":
        "Build Matplotlib and Seaborn charts on a public dataset",
    "Machine Learning":
        "Complete Scikit-learn basics: classification and regression",
    "Statistics":
        "Study mean, median, standard deviation, and basic probability",
    "Excel":
        "Practice VLOOKUP, pivot tables, and conditional formatting",
    "Python":
        "Strengthen Python: functions, OOP, file handling with real exercises",
    "DSA":
        "Solve 3 LeetCode easy problems daily — arrays and strings first",
    "OOP":
        "Build one Object-Oriented Python mini project this week",
    "DBMS":
        "Study ACID properties, normalisation, and basic SQL queries",
    "Git":
        "Create a GitHub account, push three projects, write good READMEs",
    "APIs":
        "Build one REST API endpoint using FastAPI or Flask",
    "Deployment":
        "Deploy one project on Streamlit Cloud or Render for free",
    "Model Evaluation":
        "Practice confusion matrix, precision, recall, F1 on a dataset",
}


def generate_roadmaps(
    role: str,
    missing_skills: List[str],
    projects: List[str],
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """
    Build a personalized 30/60/90-day roadmap from the student's actual gaps.

    How personalization works:
        1. Skill gap tasks — added only for skills the student is missing.
        2. Profile-aware tasks — added when CGPA, projects, coding, or
           internship status falls below the recommended threshold.
        3. The 30-day plan always ends with a concrete project to build.
        4. The 60-day and 90-day plans are role-agnostic growth steps that
           any student benefits from regardless of their target role.

    Args:
        role:           Target role string (e.g. "FinTech Analyst").
        missing_skills: List of skills the student still needs.
        projects:       List of suggested project titles for the role.
        profile:        Full profile dict for profile-aware gap detection.
                        Pass None to skip profile-aware tasks (backward-compat).

    Returns:
        Dict with keys '30_day', '60_day', '90_day', each a list of task strings.
    """
    roadmap_30: list[str] = []

    # ── Step 1: Add a task for each missing skill ─────────────────────────────
    for skill in missing_skills:
        if skill in _SKILL_TASKS:
            roadmap_30.append(_SKILL_TASKS[skill])

    # ── Step 2: Profile-aware gap detection ───────────────────────────────────
    # These tasks only appear when the student's profile reveals a weakness
    # that isn't covered by the skill gap tasks above.
    if profile is not None:
        if profile.get("cgpa", 10.0) < 7.0:
            roadmap_30.append(
                "Prioritise academic performance — target 7.0+ CGPA for placement eligibility"
            )

        if profile.get("projects", 0) < 2:
            roadmap_30.append(
                "Build at least 2 portfolio projects this month — quality matters more than quantity"
            )

        if profile.get("coding_platform_activity", 100) < 50:
            roadmap_30.append(
                "Solve 3 LeetCode / HackerRank problems daily to reach 50+ total this month"
            )

        if profile.get("internship_status", "") in ["No Internship", "Applied"]:
            roadmap_30.append(
                "Apply to 5 internships this week on Internshala or LinkedIn"
            )

    # ── Step 3: Always end with a concrete project ────────────────────────────
    project_title: str = projects[0] if projects else f"{role} starter project"
    roadmap_30.append(f"Build and publish: {project_title}")

    # ── Deduplicate while preserving order ────────────────────────────────────
    seen: set[str] = set()
    roadmap_30_clean: list[str] = []
    for item in roadmap_30:
        if item not in seen:
            seen.add(item)
            roadmap_30_clean.append(item)

    # ── 60-day: portfolio and community building ──────────────────────────────
    roadmap_60: list[str] = [
        f"Build a second portfolio project: {projects[1] if len(projects) > 1 else role + ' analysis'}",
        "Polish GitHub profile — add descriptions, screenshots, and live demo links",
        "Solve 50 more coding problems and attempt one mock test",
        "Join one community: Kaggle, GitHub Discussions, or a Discord study group",
        "Write a short blog or LinkedIn post explaining one of your projects",
    ]

    # ── 90-day: job readiness ─────────────────────────────────────────────────
    roadmap_90: list[str] = [
        "Apply to 10 companies or internships with a tailored resume",
        "Prepare for HR and technical interviews — practice STAR method answers",
        "Complete at least one mock interview with a peer or on Pramp",
        "Update LinkedIn with projects, skills, certifications, and a headline",
        "Document your preparation journey in a README or portfolio site",
    ]

    return {
        "30_day": roadmap_30_clean,
        "60_day": roadmap_60,
        "90_day": roadmap_90,
    }
