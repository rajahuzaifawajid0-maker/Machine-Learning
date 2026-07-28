# =====================================================
# STEP 1 : Import Libraries
# =====================================================

import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split



# =====================================================
# STEP 2 : Dynamic Preprocessing Function
# =====================================================

def preprocess_data(filepath):


    # Load Dataset

    df = pd.read_csv(filepath)



    # =================================================
    # Remove Duplicate Rows
    # =================================================

    df = df.drop_duplicates()



    # =================================================
    # Handle Missing Values
    # =================================================


    numerical_columns = df.select_dtypes(
        include=["int64","float64"]
    ).columns


    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns



    # Numerical Missing Values

    for col in numerical_columns:

        df[col] = df[col].fillna(
            df[col].mean()
        )



    # Categorical Missing Values

    for col in categorical_columns:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )



    # =================================================
    # Encoding Categorical Columns
    # =================================================


    encoder = LabelEncoder()


    for col in categorical_columns:

        df[col] = encoder.fit_transform(
            df[col]
        )



    # =================================================
    # Automatic Target Detection
    # Last Column = Target
    # =================================================


    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]



    # =================================================
    # Feature Scaling
    # =================================================


    scaler = StandardScaler()

    X = scaler.fit_transform(X)



    # =================================================
    # Train Test Split
    # =================================================


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.2,
        random_state=42

    )



    # Convert Processed Data

    processed_df = pd.DataFrame(
        X,
        columns=df.iloc[:, :-1].columns
    )


    processed_df["Target"] = y



    # Save CSV

    processed_df.to_csv(
        "processed/preprocessed_data.csv",
        index=False
    )



    return (
        X_train,
        X_test,
        y_train,
        y_test,
        processed_df
    )