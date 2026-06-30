"""
src/analysis.py
===============
Placement readiness analysis logic — upgraded with type hints and expanded docs.

This module solves the core business problem: convert a student profile into
a useful mentoring report using transparent, rule-based scoring.

Why rule-based (not ML) for the score?
    ML models are black boxes — a student cannot understand *why* they got a
    certain score. Rule-based scoring is transparent: every point can be
    explained in a sentence, which is more useful for mentoring.

Public API used by app.py and main.py:
    TARGET_ROLE_DATA  — dict of role → required skills / projects / resources
    analyze_student_profile(profile) → report dict
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.roadmap import generate_roadmaps


# ── Role catalogue ────────────────────────────────────────────────────────────
# Each role entry defines what skills, projects, and resources the student needs.
# Adding a new role here automatically populates the Streamlit dropdown,
# gap analysis, and roadmap generator — no other file needs to change.

TARGET_ROLE_DATA: Dict[str, Dict[str, List[str]]] = {
    "Data Analyst": {
        "skills": ["Python", "SQL", "Excel", "Pandas", "Data Visualization"],
        "projects": [
            "Sales dashboard with CSV data",
            "Student performance analysis",
            "Banking transaction dashboard",
        ],
        "resources": [
            "Kaggle Pandas micro-course",
            "SQLBolt",
            "Excel practice dashboards",
        ],
    },
    "Data Scientist": {
        "skills": ["Python", "Statistics", "Pandas", "Machine Learning", "Data Visualization"],
        "projects": [
            "Placement prediction model",
            "Customer churn prediction",
            "EDA on public health dataset",
        ],
        "resources": [
            "Kaggle Intro to ML",
            "StatQuest basics",
            "Pandas official tutorials",
        ],
    },
    "ML Engineer": {
        "skills": ["Python", "Machine Learning", "Model Evaluation", "APIs", "Deployment"],
        "projects": [
            "Streamlit ML prediction app",
            "Model comparison notebook",
            "Simple API for predictions",
        ],
        "resources": [
            "Scikit-learn user guide",
            "FastAPI beginner tutorial",
            "ML evaluation metrics notes",
        ],
    },
    "Software Developer": {
        "skills": ["Python", "DSA", "OOP", "DBMS", "Git"],
        "projects": [
            "CRUD student manager",
            "REST API mini project",
            "DSA tracker with CSV storage",
        ],
        "resources": [
            "LeetCode easy patterns",
            "GitHub Skills",
            "DBMS interview notes",
        ],
    },
    "FinTech Analyst": {
        "skills": ["SQL", "Excel", "Pandas", "Data Visualization"],
        "projects": [
            "Banking dataset analysis",
            "Loan approval dashboard",
            "UPI transaction trend analysis",
        ],
        "resources": [
            "SQL joins practice",
            "Excel pivot table practice",
            "Pandas groupby tutorial",
        ],
    },
}


# ── Skill normalization ───────────────────────────────────────────────────────

def normalize_skills(skills_text: str) -> Set[str]:
    """
    Convert a comma-separated skill string into a cleaned lowercase set.

    Why normalise?
        Students write skills in inconsistent formats: 'Python', 'python',
        'PYTHON', 'Basic Python'.  Normalising to lowercase makes comparison
        fair without penalising different capitalisation styles.

    Args:
        skills_text: Raw string from the Streamlit text area,
                     e.g. "Python, Excel, Basic SQL".

    Returns:
        Set of lowercase trimmed skill strings,
        e.g. {"python", "excel", "basic sql"}.
    """
    return {skill.strip().lower() for skill in skills_text.split(",") if skill.strip()}


def skill_is_present(required_skill: str, student_skills: Set[str]) -> bool:
    """
    Check whether a required skill exists in the student's skill set.

    Why substring matching (not exact match)?
        'Basic SQL' should count for the required skill 'SQL' because beginners
        often prefix skill names with their proficiency level.  Substring
        matching in both directions catches these cases.

    Args:
        required_skill: One skill from TARGET_ROLE_DATA, e.g. "SQL".
        student_skills: Normalised skill set from normalize_skills().

    Returns:
        True when the student has the skill or a close variant.
    """
    required_lower: str = required_skill.lower()
    return any(
        required_lower in student_skill or student_skill in required_lower
        for student_skill in student_skills
    )


# ── Score calculation ─────────────────────────────────────────────────────────

def calculate_score_breakdown(
    profile: Dict[str, Any],
    matched_skill_count: int,
    required_skill_count: int,
) -> Dict[str, float]:
    """
    Calculate five score components that add up to a maximum of 100.

    Component weights (why these numbers?):
        CGPA (20 pts)     — Academic eligibility threshold; most companies filter here.
        Skills (25 pts)   — Role alignment is the biggest placement signal.
        Projects (20 pts) — Proof of practical application beyond theory.
        Internship (10 pts)— Industry exposure often distinguishes tied candidates.
        Coding (25 pts)   — Consistent practice correlates with technical interview success.

    np.clip is used instead of if-statements to keep each component within its
    maximum even when a student overshoots (e.g. 10 projects → still max 20 pts).

    Args:
        profile:              Student profile dictionary.
        matched_skill_count:  Number of required role skills the student has.
        required_skill_count: Total required skills for the target role.

    Returns:
        Dictionary mapping component name → score contribution (float).
    """
    cgpa_score: float = np.clip((profile["cgpa"] / 10) * 20, 0, 20)
    skill_score: float = np.clip(
        (matched_skill_count / max(required_skill_count, 1)) * 25, 0, 25
    )
    project_score: float = np.clip((profile["projects"] / 4) * 20, 0, 20)
    internship_score: float = {
        "No Internship": 0.0,
        "Applied": 4.0,
        "Doing Internship": 8.0,
        "Completed Internship": 10.0,
    }.get(profile["internship_status"], 0.0)
    coding_score: float = np.clip(
        (profile["coding_platform_activity"] / 200) * 25, 0, 25
    )

    return {
        "CGPA": round(float(cgpa_score), 1),
        "Skills": round(float(skill_score), 1),
        "Projects": round(float(project_score), 1),
        "Internship": round(float(internship_score), 1),
        "Coding Practice": round(float(coding_score), 1),
    }


# ── Strength / weakness classification ───────────────────────────────────────

def find_strengths_and_weaknesses(
    profile: Dict[str, Any],
    matched_skill_count: int,
) -> tuple[List[str], List[str]]:
    """
    Classify each placement-relevant area as a strength or weakness.

    Design decision:
        Binary classification (good/bad) is simpler to act on than a spectrum.
        A student who reads 'Role skills need more alignment' knows immediately
        to add more matching skills — no interpretation needed.

    Thresholds and their rationale:
        CGPA ≥ 7.0      — Common minimum eligibility cutoff at most companies.
        Skills ≥ 3      — At least 3 of 4-5 required role skills shows alignment.
        Projects ≥ 2    — Industry expects 2+ projects from a 2nd-year student.
        Internship      — Any active exposure (doing or completed) counts.
        Coding ≥ 100    — ~100 problems distinguishes consistent from occasional practice.

    Args:
        profile:             Student profile dictionary.
        matched_skill_count: Number of role skills the student has.

    Returns:
        Tuple of (strengths list, weaknesses list).
    """
    checks: List[Tuple[str, str, bool]] = [
        (
            "CGPA is placement-friendly",
            "CGPA needs improvement for more eligibility",
            profile["cgpa"] >= 7.0,
        ),
        (
            "Role skills are aligned",
            "Role skills need more alignment",
            matched_skill_count >= 3,
        ),
        (
            "Project count is healthy",
            "Project portfolio needs more work",
            profile["projects"] >= 2,
        ),
        (
            "Internship exposure is present",
            "Internship exposure needs improvement",
            profile["internship_status"] in ["Doing Internship", "Completed Internship"],
        ),
        (
            "Coding practice is active",
            "Coding practice needs more consistency",
            profile["coding_platform_activity"] >= 100,
        ),
    ]

    strengths: List[str] = [strength for strength, _, passed in checks if passed]
    weaknesses: List[str] = [weakness for _, weakness, passed in checks if not passed]
    return strengths, weaknesses


# ── Readiness grade label ─────────────────────────────────────────────────────

def score_to_grade(score: int) -> str:
    """
    Convert a numeric readiness score to a descriptive grade label.

    Grade thresholds are set to be achievable by a 2nd-year student:
        90+ requires near-perfect CGPA, full skills, many projects, and active coding.
        75-89 is attainable with solid academic + practical preparation.
        60-74 indicates one or two important areas still need work.
        <60 means the student should focus on fundamentals before applying.

    Args:
        score: Integer readiness score (0 – 100).

    Returns:
        Grade string: 'Excellent', 'Good', 'Average', or 'Needs Improvement'.
    """
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Average"
    return "Needs Improvement"


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_student_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the complete placement mentor report from a student profile.

    This is the single function called by both app.py (Streamlit) and main.py
    (terminal demo).  It orchestrates all sub-modules and returns one dictionary
    that the UI can unpack without knowing the internal logic.

    Args:
        profile: Dictionary with keys:
            name, roll_no, cgpa, skills, projects,
            internship_status, coding_platform_activity, target_role.

    Returns:
        Report dictionary with keys:
            saved_profile, required_skills, matched_skills, missing_skills,
            score_breakdown, readiness_score, readiness_grade,
            completion_percentage, strengths, weaknesses,
            suggested_projects, learning_resources, roadmaps.
    """
    role_data: dict = TARGET_ROLE_DATA[profile["target_role"]]
    student_skills: Set[str] = normalize_skills(profile["skills"])
    required_skills: List[str] = role_data["skills"]

    matched_skills: List[str] = [
        skill for skill in required_skills
        if skill_is_present(skill, student_skills)
    ]
    missing_skills: List[str] = [
        skill for skill in required_skills
        if skill not in matched_skills
    ]

    score_breakdown: Dict[str, float] = calculate_score_breakdown(
        profile, len(matched_skills), len(required_skills)
    )
    readiness_score: int = int(round(sum(score_breakdown.values())))
    completion_percentage: int = int(
        round((len(matched_skills) / len(required_skills)) * 100)
    )
    strengths, weaknesses = find_strengths_and_weaknesses(profile, len(matched_skills))
    readiness_grade: str = score_to_grade(readiness_score)

    roadmaps: dict = generate_roadmaps(
        profile["target_role"],
        missing_skills,
        role_data["projects"],
        profile=profile,
    )

    # saved_profile is what gets written to students.csv and used by the ML module
    saved_profile: Dict[str, Any] = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": profile["name"],
        "roll_no": profile["roll_no"],
        "cgpa": profile["cgpa"],
        "skills": profile["skills"],
        "projects": profile["projects"],
        "internship_status": profile["internship_status"],
        "coding_platform_activity": profile["coding_platform_activity"],
        "target_role": profile["target_role"],
        "readiness_score": readiness_score,
        "readiness_grade": readiness_grade,
        "completion_percentage": completion_percentage,
    }

    return {
        "saved_profile": saved_profile,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "score_breakdown": score_breakdown,
        "readiness_score": readiness_score,
        "readiness_grade": readiness_grade,
        "completion_percentage": completion_percentage,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggested_projects": role_data["projects"],
        "learning_resources": role_data["resources"],
        "roadmaps": roadmaps,
    }
