# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
df = pd.read_csv("supermarket_sales.csv")

# Step 3: View the Dataset
print(df.head(10))
print(df.tail(10))

# Step 4: Check Dataset Size
print(df.shape)
print(df.columns)

# Step 5: Check Data Types
print(df.dtypes)

# Step 6: Check Missing Values
print(df.isnull().sum())

# Step 7: Fill Missing Values
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Step 8: Remove Duplicate Values
df = df.drop_duplicates()

# Step 9: Convert Date Format
df["Date"] = pd.to_datetime(df["Date"])

# Step 10: Create Month Column
df["Month"] = df["Date"].dt.month_name()

# Step 11: Create Revenue Column
df["Revenue"] = df["Total"]

# Step 12: Monthly Revenue Analysis
monthly_revenue = df.groupby("Month")["Revenue"].sum()
print(monthly_revenue)

# Step 13: Line Chart
plt.figure(figsize=(8,5))
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o')
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# Step 14: Product Line Analysis
product_sales = df.groupby("Product line")["Revenue"].sum()
print(product_sales)

# Step 15: Bar Chart
plt.figure(figsize=(10,5))
plt.bar(product_sales.index, product_sales.values)
plt.title("Revenue by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# Step 16: Payment Method Analysis
payment_counts = df["Payment"].value_counts()
print(payment_counts)

# Step 17: Pie Chart
plt.figure(figsize=(6,6))
plt.pie(payment_counts.values,
        labels=payment_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Payment Method Distribution")
plt.show()

# Step 18: Customer Type Distribution
customer_counts = df["Customer type"].value_counts()
print(customer_counts)

# Step 19: Customer Type Bar Chart
plt.figure(figsize=(6,5))
plt.bar(customer_counts.index, customer_counts.values)
plt.title("Customer Type Distribution")
plt.xlabel("Customer Type")
plt.ylabel("Count")
plt.show()

# Step 20: Quantity Histogram
plt.figure(figsize=(6,5))
plt.hist(df["Quantity"], bins=10)
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

# Step 21: Scatter Plot
plt.figure(figsize=(6,5))
plt.scatter(df["Quantity"], df["Revenue"])
plt.title("Quantity vs Revenue")
plt.xlabel("Quantity")
plt.ylabel("Revenue")
plt.show()