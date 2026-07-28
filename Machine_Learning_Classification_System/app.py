# =====================================================
# STEP 1 : Import Libraries
# =====================================================

from flask import Flask, render_template, request,send_file
from preprocessing import preprocess_data
from model_training import train_model, models
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# STEP 2 : Create Flask Application
# =====================================================

app = Flask(__name__)


# =====================================================
# STEP 3 : Create Required Folders
# =====================================================

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_FOLDER = "models"


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



# =====================================================
# STEP 4 : Home Page
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")



# =====================================================
# STEP 5 : Upload CSV File
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]


    if file.filename == "":

        return "No File Selected"


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )


    file.save(filepath)



    # Read Dataset

    df = pd.read_csv(filepath)


    rows = df.shape[0]

    columns = df.shape[1]


    preview = df.head().to_html()



    return f"""

    <h1>CSV Uploaded Successfully</h1>

    <hr>

    <h3>File Name : {file.filename}</h3>

    <h3>Total Rows : {rows}</h3>

    <h3>Total Columns : {columns}</h3>


    <h2>Dataset Preview</h2>

    {preview}


    <br>

    <a href="/">Go Back</a>

    """



# =====================================================
# STEP 6 : Run Flask move to last lines
# =====================================================


# =====================================================
# STEP 7 : Dynamic Preprocessing
# =====================================================


@app.route("/preprocess")
def preprocess():

    filepath = "uploads/" + os.listdir("uploads")[0]


    X_train, X_test, y_train, y_test, df = preprocess_data(filepath)



    return """

    <h1>
    Dynamic Preprocessing Completed
    </h1>


    <hr>


    <h3>
    Missing Values Handled
    </h3>


    <h3>
    Categorical Encoding Completed
    </h3>


    <h3>
    Feature Scaling Completed
    </h3>


    <h3>
    Duplicate Rows Removed
    </h3>


    <h3>
    Train Test Split Completed
    </h3>


    <br>


    <a href="/download">
    Download Preprocessed Dataset
    </a>

    """
# =====================================================
# STEP 8 : Download Processed Dataset
# =====================================================


@app.route("/download")
def download():


    file_path = "processed/preprocessed_data.csv"


    return send_file(
        file_path,
        as_attachment=True
    )
# =====================================================
#STEO 9:
# =====================================================
@app.route("/train", methods=["POST"])
def train():
    model_name = request.form["model"]

    filepath = "uploads/" + os.listdir("uploads")[0]

    X_train, X_test, y_train, y_test, df = preprocess_data(filepath)

    result = train_model(

        model_name,

        X_train,

        X_test,

        y_train,

        y_test

    )

    return f"""

     <h1>
     Model Evaluation Result
     </h1>

     <h2>
     Selected Model:
     {model_name}
     </h2>


     <h3>
     Accuracy:
     {result["accuracy"] * 100:.2f} %
     </h3>


     <h3>
     Precision:
     {result["precision"] * 100:.2f} %
     </h3>


     <h3>
     Recall:
     {result["recall"] * 100:.2f} %
     </h3>


     <h3>
     F1 Score:
     {result["f1"] * 100:.2f} %
     </h3>


     <h2>
     Confusion Matrix
     </h2>


     <img src="/static/confusion_matrix.png"
     width="400">


     <h2>
     Classification Report
     </h2>


     <pre>
     {result["report"]}
     </pre>

     """
if __name__ == "__main__":

    app.run(debug=True,port=800)