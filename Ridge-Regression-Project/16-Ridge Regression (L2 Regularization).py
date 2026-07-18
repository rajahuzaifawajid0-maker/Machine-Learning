# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read Dataset
df = pd.read_csv("Employers_data.csv")
print(df.head())
print(df.tail())
print(df.describe())



# Select Features (Independent Variable)
X = df[['Experience_Years']]

# Select Target (Dependent Variable)
y = df['Salary']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Ridge Model
model = Ridge(alpha=1.0)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R2 Score :", r2)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(df["Experience_Years"], df["Salary"])
plt.title("Experience vs Salary")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))

df_sorted = df.sort_values("Experience_Years")

plt.plot(df_sorted["Experience_Years"], df_sorted["Salary"], marker='o')

plt.title("Salary Growth with Experience")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))

plt.hist(df["Salary"], bins=10)

plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))

plt.hist(df["Experience_Years"], bins=10)

plt.title("Experience Distribution")
plt.xlabel("Experience (Years)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))

plt.scatter(df["Experience_Years"], df["Salary"], label="Actual Data")

X_sorted = X.sort_values(by="Experience_Years")

plt.plot(
    X_sorted,
    model.predict(X_sorted),
    linewidth=3,
    label="Ridge Regression"
)

plt.title("Ridge Regression")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title("Actual vs Predicted Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.grid(True)
plt.show()


# Count each education level
education_counts = df["Education_Level"].value_counts()

# Create Pie Chart
plt.figure(figsize=(8,8))

plt.pie(
    education_counts,
    labels=education_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    shadow=True
)

plt.title("Education Level Distribution")
plt.axis("equal")   # Makes the pie chart circular

plt.show()
Gender_counts = df["Gender"].value_counts()

plt.figure(figsize=(8,8))
plt.pie(
    Gender_counts,
    labels=Gender_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    shadow=True
)
plt.title("Gender Distribution")
plt.axis("equal")
plt.show()