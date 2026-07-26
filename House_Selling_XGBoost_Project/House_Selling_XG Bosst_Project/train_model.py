# =====================================================
# STEP 1 : Import Libraries
# =====================================================

# Data ko read aur process karne ke liye
import pandas as pd

# Mathematical operations ke liye
import numpy as np

# Dataset ko Train aur Test mein divide karne ke liye
from sklearn.model_selection import train_test_split
#Label Encoding
from sklearn.preprocessing import LabelEncoder
# XGBoost Regression Model import
from xgboost import XGBRegressor

# Model ko save aur load karne ke liye
import os
import joblib
# Model Evaluation Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# STEP 2 : Load Dataset
# =====================================================

# CSV file ko read karna
df = pd.read_csv("dataset/House Price Prediction Dataset.csv")

# First 5 rows dekhna
print(df.head())
# =====================================================
# STEP 4 : Shape of Dataset
# =====================================================

print(df.shape)
# =====================================================
# STEP 5 : Column Names
# =====================================================

print(df.columns)
# =====================================================
# STEP 6 : Statistical Summary
# =====================================================

print(df.describe())
# =====================================================
# STEP 7 : Check Missing Values
# =====================================================

print(df.isnull().sum())
# =====================================================
# Encode Categorical Columns
# =====================================================

# Label Encoder object
le = LabelEncoder()

# String columns ko number mein convert karna
df["Location"] = le.fit_transform(df["Location"])

df["Condition"] = le.fit_transform(df["Condition"])

df["Garage"] = le.fit_transform(df["Garage"])
print(df.dtypes)
# =====================================================
# STEP 9 : Select Features and Target
# =====================================================

# Target column ko y mein store karo
y = df["Price"]

# Price ke ilawa baaki sab columns X mein store karo
X = df.drop(["Id", "Price"], axis=1)
print(X.columns)

# Features aur Target ka size check karo
print("Features Shape :", X.shape)
print("Target Shape   :", y.shape)
# =====================================================
# STEP 10 : Split Dataset into Training and Testing
# =====================================================

# Dataset ko Training aur Testing data mein divide karna
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Shapes check karna
print("X_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)
# =====================================================
# STEP 11 : Create XGBoost Regression Model
# =====================================================

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

print(model)

# =====================================================
# STEP 12 : Train the XGBoost Model
# =====================================================

# Model ko Training Data se train karna
model.fit(X_train, y_train)

print("========================================")
print("Model Training Completed Successfully")
print("========================================")

# =====================================================
# STEP 13 : Predict House Prices
# =====================================================

# Test Data par Prediction karna
y_pred = model.predict(X_test)

print("First 10 Predictions")
print(y_pred[:10])

# =====================================================
# STEP 14 : Evaluate Model
# =====================================================

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# R2 Score
r2 = r2_score(y_test, y_pred)


print("===================================")
print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("R2 Score            :", r2)
print("===================================")
# =====================================================
# STEP 15 : Save Model
# =====================================================

# models folder ke andar model save karna
joblib.dump(model, "models/xgboost_model.pkl")

print("===================================")
print("Model Saved Successfully")
print("===================================")
