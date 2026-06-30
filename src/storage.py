"""
src/storage.py
==============
CSV storage helpers using Pandas — upgraded with type hints and expanded docs.

Why CSV instead of a database?
    CSV is human-readable, opens in Excel for verification, requires zero
    server setup, and is sufficient for a portfolio project with <1000 records.
    In production this module could be swapped for SQLite by changing only the
    two functions below — the rest of the codebase stays unchanged.

Public API:
    load_student_profiles() → pd.DataFrame
    save_student_profile(profile: dict) → pd.DataFrame
"""

from pathlib import Path
from typing import List

import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
STUDENT_CSV: Path = DATA_DIR / "students.csv"

# Column order matches the saved_profile dict produced by analysis.py.
# Keeping the column list here (not scattered across the codebase) means
# adding a new field only requires editing this one list.
STUDENT_COLUMNS: List[str] = [
    "saved_at",
    "name",
    "roll_no",
    "cgpa",
    "skills",
    "projects",
    "internship_status",
    "coding_platform_activity",
    "target_role",
    "readiness_score",
    "completion_percentage",
]


def ensure_data_file() -> Path:
    """
    Create the data directory and CSV file with headers if they do not exist.

    Why this function exists:
        The app should work on a fresh clone without the student needing to
        manually create directories or files.  Running `streamlit run app.py`
        for the first time automatically creates `data/students.csv`.

    Returns:
        Absolute path to the students CSV file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STUDENT_CSV.exists():
        pd.DataFrame(columns=STUDENT_COLUMNS).to_csv(STUDENT_CSV, index=False)
    return STUDENT_CSV


def load_student_profiles() -> pd.DataFrame:
    """
    Read all saved student records from the CSV file.

    Why Pandas here?
        Pandas handles missing columns gracefully, makes the data easy to
        display in Streamlit with `st.dataframe`, and allows future filtering
        or analysis with a single line of code.

    Returns:
        DataFrame of saved profiles (empty DataFrame if no profiles yet).
    """
    csv_path: Path = ensure_data_file()
    return pd.read_csv(csv_path)


def save_student_profile(profile: dict) -> pd.DataFrame:
    """
    Save or update one student profile in the CSV file.

    Update logic:
        If the same roll number already exists, the old row is removed before
        the new one is appended.  This prevents duplicate entries when a
        student submits the form multiple times with updated data.

    Args:
        profile: Dictionary produced by analyze_student_profile() in analysis.py.
                 Must contain a 'roll_no' key.

    Returns:
        Updated DataFrame reflecting the saved state of the CSV.
    """
    df: pd.DataFrame = load_student_profiles()

    # Build a single-row DataFrame from the profile dict.
    # Only columns in STUDENT_COLUMNS are kept — extra keys (like roadmaps)
    # are ignored so the CSV stays clean.
    row_data: dict = {col: profile.get(col) for col in STUDENT_COLUMNS}
    new_row: pd.DataFrame = pd.DataFrame([row_data], columns=STUDENT_COLUMNS)

    # Remove the existing row for this roll number to avoid duplicates
    if not df.empty and str(profile["roll_no"]) in df["roll_no"].astype(str).values:
        df = df[df["roll_no"].astype(str) != str(profile["roll_no"])]

    updated_df: pd.DataFrame = (
        new_row if df.empty
        else pd.concat([df, new_row], ignore_index=True)
    )
    updated_df.to_csv(STUDENT_CSV, index=False)
    return updated_df
