# =====================================================
# STEP 1 : Import Required Libraries
# =====================================================

# Flask Library
from flask import Flask, render_template, request

# Model Load Karne Ke Liye
import joblib

# Numpy Array Banane Ke Liye
import numpy as np
# =====================================================
# STEP 2 : Create Flask App
# =====================================================

app = Flask(__name__)
# =====================================================
# STEP 3 : Load Trained Model
# =====================================================

model = joblib.load("models/xgboost_model.pkl")
# =====================================================
# STEP 4 : Home Page
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")
# =====================================================
# =====================================================
# STEP 5 : Prediction Route
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Form se data lena
    area = float(request.form["Area"])
    bedrooms = int(request.form["Bedrooms"])
    bathrooms = int(request.form["Bathrooms"])
    floors = int(request.form["Floors"])
    year_built = int(request.form["YearBuilt"])
    location = int(request.form["Location"])
    condition = int(request.form["Condition"])
    garage = int(request.form["Garage"])

    # Model ke liye input array banana
    input_data = np.array([[

        area,
        bedrooms,
        bathrooms,
        floors,
        year_built,
        location,
        condition,
        garage
    ]])
    print("Input Shape:", input_data.shape)
    print("Input Data:", input_data)

    # Prediction karna
    prediction = model.predict(input_data)

    # Result page open karna
    return render_template(
        "result.html",
        prediction=round(prediction[0], 2)
    )
# STEP 6 : Run Flask Application
# =====================================================

if __name__ == "__main__":
    app.run(debug=True, port=8000)