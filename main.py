"""Command-line starter for Placement Mentor Lite.

Demonstrates that core logic works without Streamlit.
Run: python3 main.py
"""

from src.analysis import analyze_student_profile
from src.ml_model import predict_placement_probability, train_placement_model
from src.storage import save_student_profile


def main():
    demo_profile = {
        "name": "Aaryan",
        "roll_no": "AIDS042",
        "cgpa": 7.2,
        "skills": "Python, Excel, Basic SQL",
        "projects": 2,
        "internship_status": "Doing Internship",
        "coding_platform_activity": 75,
        "target_role": "FinTech Analyst",
    }

    report = analyze_student_profile(demo_profile)
    model_bundle = train_placement_model()
    placement_probability = predict_placement_probability(model_bundle, report["saved_profile"])
    save_student_profile(report["saved_profile"])

    print("\n" + "=" * 45)
    print("  Placement Mentor Lite - Demo Report")
    print("=" * 45)
    print(f"  Student     : {report['saved_profile']['name']}")
    print(f"  Target Role : {report['saved_profile']['target_role']}")
    print(f"  Score       : {report['readiness_score']}/100  ({report['readiness_grade']})")
    print(f"  Completion  : {report['completion_percentage']}%")
    print(f"  ML Prob     : {placement_probability}%")

    print("\n--- Model Comparison ---")
    for model_name, acc in model_bundle["model_accuracies"].items():
        marker = " ◀ best" if model_name == model_bundle["model_name"] else ""
        print(f"  {model_name:<22}: {acc}%{marker}")

    print("\n--- Feature Importance ---")
    for feat, imp in model_bundle["feature_importance"].items():
        bar = "█" * int(imp / 4)
        print(f"  {feat:<20}: {imp:>5}%  {bar}")

    print("\n--- Strengths ---")
    for s in report["strengths"]:
        print(f"  + {s}")

    print("\n--- Weaknesses ---")
    for w in report["weaknesses"]:
        print(f"  - {w}")

    print("\n--- Missing Skills ---")
    for skill in report["missing_skills"]:
        print(f"  ✗ {skill}")

    print("\n--- Personalized 30-Day Roadmap ---")
    for i, task in enumerate(report["roadmaps"]["30_day"], start=1):
        print(f"  {i}. {task}")

    print("\n" + "=" * 45 + "\n")


if __name__ == "__main__":
    main()
