"""
scripts/generate_dataset.py
============================
Generates a realistic synthetic placement dataset with 200 student records.

Why this script exists:
    Real placement datasets are confidential. This script creates a believable
    synthetic dataset that lets the ML model learn from feature *combinations*
    rather than a single factor, making it more honest and educational.

How to run:
    python scripts/generate_dataset.py

Output:
    data/placement_sample.csv  (200 rows, 6 columns)

Columns (names must NOT change — ml_model.py depends on them):
    cgpa                    — Grade point average out of 10
    projects                — Number of completed projects
    coding_platform_activity — Total problems solved on coding platforms
    internship_score        — 0.0 / 0.3 / 0.7 / 1.0 (No / Applied / Doing / Completed)
    skill_match_percentage  — Percentage of target-role skills the student has
    placed                  — Target label: 1 = placed, 0 = not placed
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd


# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Output path ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_CSV: Path = PROJECT_ROOT / "data" / "placement_sample.csv"


# ── Helper: clip to valid range ───────────────────────────────────────────────
def _clip(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high (inclusive)."""
    return float(np.clip(value, low, high))


# ── Student group generators ──────────────────────────────────────────────────

def _excellent_student() -> dict:
    """
    40 records — Strong across all areas.
    Placed probability: ~90 %.
    Realistic exception: occasionally a high-CGPA student still misses placement
    because of poor coding practice alone.
    """
    cgpa = _clip(np.random.normal(8.8, 0.4), 7.8, 10.0)
    projects = random.randint(3, 6)
    coding = _clip(np.random.normal(200, 30), 120, 300)
    internship = random.choices([0.3, 0.7, 1.0], weights=[10, 35, 55])[0]
    skill_match = _clip(np.random.normal(85, 8), 65, 100)

    # Realistic exception: great CGPA, barely any coding → not placed
    if random.random() < 0.08:
        coding = _clip(np.random.normal(40, 10), 10, 60)
        placed = 0
    else:
        placed = 1 if random.random() < 0.90 else 0

    return dict(cgpa=round(cgpa, 2), projects=projects,
                coding_platform_activity=int(coding),
                internship_score=internship,
                skill_match_percentage=round(skill_match, 1),
                placed=placed)


def _good_student() -> dict:
    """
    50 records — Above-average profile.
    Placed probability: ~70 %.
    Realistic exception: solid internship but weak coding → sometimes not placed.
    """
    cgpa = _clip(np.random.normal(7.8, 0.5), 6.8, 9.0)
    projects = random.randint(2, 4)
    coding = _clip(np.random.normal(130, 35), 60, 250)
    internship = random.choices([0.0, 0.3, 0.7, 1.0], weights=[15, 20, 35, 30])[0]
    skill_match = _clip(np.random.normal(70, 10), 50, 90)

    # Realistic exception: internship done, but coding is very low → miss
    if internship >= 0.7 and random.random() < 0.12:
        coding = _clip(np.random.normal(30, 10), 5, 55)
        placed = 0
    else:
        placed = 1 if random.random() < 0.70 else 0

    return dict(cgpa=round(cgpa, 2), projects=projects,
                coding_platform_activity=int(coding),
                internship_score=internship,
                skill_match_percentage=round(skill_match, 1),
                placed=placed)


def _average_student() -> dict:
    """
    50 records — Mid-range profile.
    Placed probability: ~45 %.
    Realistic exception: strong project count but average CGPA → placed.
    """
    cgpa = _clip(np.random.normal(6.9, 0.5), 5.8, 8.0)
    projects = random.randint(1, 3)
    coding = _clip(np.random.normal(80, 30), 20, 180)
    internship = random.choices([0.0, 0.3, 0.7, 1.0], weights=[30, 30, 25, 15])[0]
    skill_match = _clip(np.random.normal(55, 12), 30, 80)

    # Realistic exception: many projects despite average CGPA → placed
    if projects >= 3 and random.random() < 0.30:
        placed = 1
    else:
        placed = 1 if random.random() < 0.45 else 0

    return dict(cgpa=round(cgpa, 2), projects=projects,
                coding_platform_activity=int(coding),
                internship_score=internship,
                skill_match_percentage=round(skill_match, 1),
                placed=placed)


def _weak_student() -> dict:
    """
    40 records — Below-average across most areas.
    Placed probability: ~15 %.
    Realistic exception: very high coding activity despite weak CGPA → sometimes placed.
    """
    cgpa = _clip(np.random.normal(5.8, 0.6), 4.0, 7.0)
    projects = random.randint(0, 2)
    coding = _clip(np.random.normal(35, 20), 0, 120)
    internship = random.choices([0.0, 0.3, 0.7], weights=[60, 30, 10])[0]
    skill_match = _clip(np.random.normal(35, 12), 10, 60)

    # Realistic exception: excellent coding even with low CGPA → placed
    if coding > 150 and random.random() < 0.35:
        placed = 1
    else:
        placed = 1 if random.random() < 0.15 else 0

    return dict(cgpa=round(cgpa, 2), projects=projects,
                coding_platform_activity=int(coding),
                internship_score=internship,
                skill_match_percentage=round(skill_match, 1),
                placed=placed)


def _borderline_student() -> dict:
    """
    20 records — Mixed profile: good in some areas, weak in others.
    Placed probability: ~50 % (pure coin flip — forces model to learn combinations).
    Examples:
        - High coding but no internship
        - Internship done but CGPA is borderline
    """
    cgpa = _clip(np.random.normal(7.0, 0.4), 6.0, 8.0)
    projects = random.randint(1, 3)

    # Mix: either great coding + no internship, or internship + poor coding
    if random.random() < 0.5:
        # Excellent coder, zero internship exposure
        coding = _clip(np.random.normal(180, 20), 140, 250)
        internship = 0.0
    else:
        # Did internship but barely practiced coding
        coding = _clip(np.random.normal(25, 10), 5, 50)
        internship = random.choice([0.7, 1.0])

    skill_match = _clip(np.random.normal(60, 15), 35, 85)
    placed = 1 if random.random() < 0.50 else 0

    return dict(cgpa=round(cgpa, 2), projects=projects,
                coding_platform_activity=int(coding),
                internship_score=internship,
                skill_match_percentage=round(skill_match, 1),
                placed=placed)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dataset() -> pd.DataFrame:
    """
    Build the full 200-record dataset from the five student groups.

    Returns:
        DataFrame with exactly 200 rows and 6 columns.
    """
    records: list[dict] = []

    # Build each group
    for _ in range(40):
        records.append(_excellent_student())
    for _ in range(50):
        records.append(_good_student())
    for _ in range(50):
        records.append(_average_student())
    for _ in range(40):
        records.append(_weak_student())
    for _ in range(20):
        records.append(_borderline_student())

    # Shuffle so groups are not clustered together (better for train/test split)
    random.shuffle(records)

    df = pd.DataFrame(records, columns=[
        "cgpa",
        "projects",
        "coding_platform_activity",
        "internship_score",
        "skill_match_percentage",
        "placed",
    ])

    return df


def main() -> None:
    """Entry point: generate dataset, save to CSV, and print a summary."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(OUTPUT_CSV, index=False)

    placed_count: int = int(df["placed"].sum())
    not_placed_count: int = len(df) - placed_count

    print("=" * 50)
    print("  Dataset Generation Complete")
    print("=" * 50)
    print(f"  Total records     : {len(df)}")
    print(f"  Placed            : {placed_count} ({placed_count / len(df) * 100:.1f}%)")
    print(f"  Not placed        : {not_placed_count} ({not_placed_count / len(df) * 100:.1f}%)")
    print(f"  Saved to          : {OUTPUT_CSV}")
    print()
    print("  Feature summary:")
    print(df.describe().round(2).to_string())
    print("=" * 50)


if __name__ == "__main__":
    main()
