# =====================================================
# STEP 1 : Import Libraries
# =====================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import joblib
import seaborn as sns

 

# =====================================================
# STEP 2 : Model Dictionary
# =====================================================

models = {

    "Logistic Regression": LogisticRegression(),

    "KNN": KNeighborsClassifier(),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "SVM": SVC(),

    "Naive Bayes": GaussianNB(),

    "Gradient Boosting": GradientBoostingClassifier(),

    "AdaBoost": AdaBoostClassifier(),

    "Extra Trees": ExtraTreesClassifier(),

    "XGBoost": XGBClassifier()

}



# =====================================================
# STEP 3 : Train Model Function
# =====================================================

def train_model(
        model_name,
        X_train,
        X_test,
        y_train,
        y_test
):

    model = models[model_name]


    # Train Model

    model.fit(
        X_train,
        y_train
    )


    # Prediction

    predictions = model.predict(
        X_test
    )


    # Evaluation

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )


    # =====================================================
    # Confusion Matrix Graph
    # =====================================================

    plt.figure(figsize=(5, 4))

    plt.imshow(cm)

    plt.colorbar()

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )

    plt.savefig(
        "static/confusion_matrix.png"
    )

    plt.close()



    # =====================================================
    # Save Model
    # =====================================================

    joblib.dump(
        model,
        "models/model.pkl"
    )


    return {

        "model": model,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "confusion_matrix": cm,

        "report": report

    }