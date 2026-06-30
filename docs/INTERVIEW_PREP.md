# Interview Preparation Mode

## Project Pitch

"Placement Mentor Lite is a Python and Streamlit dashboard that helps students understand their placement readiness. It takes profile inputs like CGPA, skills, projects, internship status, coding activity, and target role. Then it calculates a readiness score, identifies skill gaps, recommends resources and projects, generates a 30/60/90-day roadmap, stores records in CSV using Pandas, and predicts placement probability using Logistic Regression."

## How To Explain The Architecture

"I separated the project into modules. `app.py` handles the dashboard, `analysis.py` handles readiness logic, `roadmap.py` creates improvement plans, `storage.py` manages CSV files with Pandas, and `ml_model.py` handles the machine learning prediction. This makes the code easier to test, explain, and improve."

## Questions And Sample Answers

Q: What problem does your project solve?

A: "Many students do not know what to improve for placements. My project gives them a simple readiness report and a practical roadmap based on their target role."

Q: Where did you use NumPy?

A: "I used NumPy in score calculation and ML feature preparation. For example, `np.clip` keeps score values inside valid limits, and NumPy arrays are used for model prediction input."

Q: Where did you use Pandas?

A: "I used Pandas to read and write CSV files, display saved profiles, and load sample placement data for the ML model."

Q: Why Logistic Regression?

A: "Placement prediction is a binary classification problem: placed or not placed. Logistic Regression is simple, explainable, and gives probability, which is suitable for a beginner-friendly ML module."

Q: What are the limitations?

A: "The ML model uses sample data, so it is not a real hiring predictor. The score weights are rule-based and should be improved with real placement data. In future, I would add authentication, real datasets, and a database."

Q: How can this project be extended?

A: "I can add resume upload, weekly progress tracking, admin dashboard for faculty, SQLite database, real placement history, and personalized resource links."
