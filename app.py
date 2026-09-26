from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/scholarship_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        maths = float(request.form["maths"])
        sinhala = float(request.form["sinhala"])
        english = float(request.form["english"])
        general_knowledge = float(request.form["general_knowledge"])
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_exam = float(request.form["previous_exam"])

        student = pd.DataFrame([{
            "maths_marks": maths,
            "sinhala_marks": sinhala,
            "english_marks": english,
            "general_knowledge": general_knowledge,
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_exam_marks": previous_exam
        }])

        prediction = model.predict(student)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)