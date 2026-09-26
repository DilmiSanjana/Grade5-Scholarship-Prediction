import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/scholarship_model.pkl")

print("======================================")
print(" Grade 5 Scholarship Prediction System")
print("======================================")

# Get student information
maths = float(input("Enter Maths marks: "))
sinhala = float(input("Enter Sinhala marks: "))
english = float(input("Enter English marks: "))
general_knowledge = float(input("Enter General Knowledge marks: "))
study_hours = float(input("Enter Study Hours per Day: "))
attendance = float(input("Enter Attendance percentage: "))
previous_exam = float(input("Enter Previous Exam marks: "))

# Create input data
student = pd.DataFrame([{
    "maths_marks": maths,
    "sinhala_marks": sinhala,
    "english_marks": english,
    "general_knowledge": general_knowledge,
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_exam_marks": previous_exam
}])

# Make prediction
prediction = model.predict(student)[0]

print("\n======================================")
print("Prediction:", prediction)
print("======================================")