# ==========================================
# STEP 1 : Import Libraries
# ==========================================

import pandas as pd
import numpy as np

# Train Test Split
from sklearn.model_selection import train_test_split

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Save Model
import joblib
# ==========================================
# STEP 2 : Load Dataset
# ==========================================

# Read Dataset
df = pd.read_csv("dataset/Titanic-Dataset.csv")

# Display First 5 Rows
print(df.head())
# ==========================================
# STEP 3 : Dataset Information
# ==========================================

# Last 5 Rows
print("\n========== Last 5 Rows ==========")
print(df.tail())

# Shape
print("\n========== Shape ==========")
print(df.shape)

# Column Names
print("\n========== Columns ==========")
print(df.columns)

# Dataset Information
print("\n========== Info ==========")
print(df.info())

# Statistical Summary
print("\n========== Describe ==========")
print(df.describe())
# ==========================================
# STEP 4 : Check Missing Values
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())
# ==========================================
# STEP 5 : Handle Missing Values
# ==========================================

# Age Column (Numerical)
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Embarked Column (Categorical)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin Column (Too Many Missing Values)
df.drop("Cabin", axis=1, inplace=True)

# Check Again
print("\nMissing Values After Handling")
print(df.isnull().sum())
# ==========================================
# STEP 6 : Check Duplicate Values
# ==========================================

print("\nDuplicate Values")
print(df.duplicated().sum())
# ==========================================
# STEP 7 : Remove Duplicate Values
# ==========================================

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates")
print(df.shape)
print("\nCurrent Dataset Shape")
print(df.shape)
# ==========================================
# STEP 9 : Label Encoding
# ==========================================

# Create Label Encoder Object
encoder = LabelEncoder()

# Convert Gender into Numbers
df["Sex"] = encoder.fit_transform(df["Sex"])

# Convert Embarked into Numbers
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# Check Dataset
print("\nDataset After Label Encoding")
print(df.head())
# ==========================================
# STEP 9.5 : Drop Unnecessary Columns
# ==========================================

df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print("\nColumns After Dropping")
print(df.columns)
print("\nData Types")
print(df.dtypes)
# ==========================================
# STEP 10 : Feature Selection
# ==========================================

# Input Features (X)
X = df.drop("Survived", axis=1)

# Output / Target (y)
y = df["Survived"]

print("\nInput Features (X)")
print(X.head())

print("\nTarget (y)")
print(y.head())
# ==========================================
# STEP 11 : Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape")
print(X_train.shape)

print("\nTesting Data Shape")
print(X_test.shape)
# ==========================================
# STEP 12 : Create Naive Bayes Model
# ==========================================

model = GaussianNB()

print("\nNaive Bayes Model Created Successfully")
model = GaussianNB()
# ==========================================
# STEP 13 : Train Naive Bayes Model
# ==========================================

# Train the Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully")
# ==========================================
# STEP 14 : Prediction
# ==========================================

# Predict Test Data
y_pred = model.predict(X_test)

print("\nPrediction Completed")
model.predict(X_test)
# ==========================================
# STEP 15 : Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")
print(accuracy)
# ==========================================
# STEP 16 : Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)
# ==========================================
# STEP 17 : Classification Report
# ==========================================

print("\nClassification Report")

print(classification_report(y_test, y_pred))
# ==========================================
# STEP 18 : Save Model
# ==========================================

# Save Naive Bayes Model
joblib.dump(model, "models/model.pkl")

print("\nModel Saved Successfully")
joblib.dump(model, "models/model.pkl")
# ==========================================
# STEP 19 : Save Label Encoder
# ==========================================

joblib.dump(encoder, "models/label_encoder.pkl")

print("Label Encoder Saved Successfully")
# ==========================================
# STEP 20 : Load Saved Model
# ==========================================

loaded_model = joblib.load("models/model.pkl")

print("\nModel Loaded Successfully")
# ==========================================
# STEP 21 : Predict New Passenger
# ==========================================

new_passenger = [[
    3,      # Pclass
    1,      # Sex (Male)
    25,     # Age
    0,      # SibSp
    0,      # Parch
    7.25,   # Fare
    2       # Embarked
]]

prediction = loaded_model.predict(new_passenger)

print("\nPrediction")

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")
