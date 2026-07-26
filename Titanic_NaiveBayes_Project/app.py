# ==========================================
# STEP 1 : Import Libraries
# ==========================================

# Flask Framework Import
from flask import Flask, render_template, request

# Load Saved Machine Learning Model
import joblib

# Create DataFrame
import pandas as pd


# ==========================================
# STEP 2 : Create Flask Application
# ==========================================

# Flask App Object
app = Flask(__name__)


# ==========================================
# STEP 3 : Load Trained Model
# ==========================================

# Load Saved Naive Bayes Model
model = joblib.load("models/model.pkl")


# ==========================================
# STEP 4 : Home Page Route
# ==========================================

# Home Page
@app.route("/")
def home():

    # Open index.html Page
    return render_template("index.html")


# ==========================================
# STEP 5 : Prediction Route
# ==========================================

# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():

    # Get Passenger Class
    pclass = int(request.form["Pclass"])

    # Get Passenger Gender
    sex = int(request.form["Sex"])

    # Get Passenger Age
    age = float(request.form["Age"])

    # Get Number of Siblings / Spouse
    sibsp = int(request.form["SibSp"])

    # Get Number of Parents / Children
    parch = int(request.form["Parch"])

    # Get Ticket Fare
    fare = float(request.form["Fare"])

    # Get Embarked Port
    embarked = int(request.form["Embarked"])

    # Create DataFrame for Prediction
    data = pd.DataFrame({

        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]

    })

    # Predict Passenger Survival
    prediction = model.predict(data)

    # Predict Probability
    probability = model.predict_proba(data)

    # Calculate Confidence Percentage
    confidence = round(max(probability[0]) * 100, 2)

    # Check Prediction Result
    if prediction[0] == 1:
        result = "Passenger Survived"

    else:
        result = "Passenger Did Not Survive"

    # Open Result Page
    return render_template(

        "result.html",
        prediction=result,
        confidence=confidence

    )


# ==========================================
# STEP 6 : Run Flask Application
# ==========================================

if __name__ == "__main__":

    # Run Flask Server
    app.run(debug=True, port=8000)