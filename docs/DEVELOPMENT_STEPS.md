# Development Steps

This file explains the project in the same order you can present it during an internship review.

## Step 1: Student Profile Input

Concept: collect measurable details such as CGPA, skills, projects, internship status, coding activity, and target role.

Code location: `app.py`

Business logic: the dashboard cannot recommend anything useful until it knows the student's current status and goal.

Interview explanation: "I designed the input form around placement factors that students can actually improve, such as skills, projects, coding practice, and internship exposure."

Common question: Why did you choose these inputs?

Sample answer: "I selected inputs that are realistic for a second-year student and useful for mentoring. CGPA shows academic eligibility, skills and projects show practical readiness, internship status shows exposure, and coding activity shows consistency."

## Step 2: Readiness Analysis

Concept: convert profile data into a score out of 100, strengths, weaknesses, and missing skills.

Code location: `src/analysis.py`

Business logic: students need prioritization. A score alone is not enough, so the app also explains what is strong and what is weak.

Interview explanation: "I used a rule-based scoring system because it is transparent, easy to debug, and suitable for a beginner portfolio project."

Common question: Is your score a real placement guarantee?

Sample answer: "No. It is a mentoring indicator. It helps compare current preparation against role expectations, but final placement depends on interviews, company criteria, and communication."

## Step 3: Roadmap Generator

Concept: convert missing skills into 30-day, 60-day, and 90-day action plans.

Code location: `src/roadmap.py`

Business logic: the project becomes useful when it tells the student exactly what to do next.

Interview explanation: "The roadmap is generated from the student's missing skills and target role, so the output is personalized instead of generic."

Common question: Why not generate a fixed roadmap?

Sample answer: "A fixed roadmap ignores the student's current level. My roadmap changes based on missing skills, which makes it more practical."

## Step 4: CSV Storage With Pandas

Concept: save and read profiles from `data/students.csv`.

Code location: `src/storage.py`

Business logic: CSV storage is enough for a beginner project and can be opened in Excel for verification.

Interview explanation: "I used Pandas because it can read, update, and display tabular student records with very little code."

Common question: Why not use a database?

Sample answer: "For this version, CSV is simpler and fits the 45-day scope. In future scope, I can replace it with SQLite or PostgreSQL."

## Step 5: Machine Learning Module

Concept: train Logistic Regression on sample placement data and predict placement probability.

Code location: `src/ml_model.py`

Business logic: ML adds a data-driven estimate, while the rule-based score remains explainable.

Interview explanation: "I used Logistic Regression because it is beginner-friendly and directly gives a probability for binary outcomes like placed or not placed."

Common question: Is the sample model production-ready?

Sample answer: "No. It is an educational model trained on sample data. For production, I would need real historical placement data, better validation, and bias checks."

## Step 6: Dashboard And Charts

Concept: show metrics, gap analysis, roadmaps, CSV records, and Matplotlib charts in Streamlit.

Code location: `app.py`

Business logic: a dashboard makes the project easier for students and faculty to use compared to terminal output.

Interview explanation: "I converted the project from Tkinter to Streamlit because Streamlit is faster for data dashboards and integrates well with Pandas and Matplotlib."
