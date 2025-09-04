import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")

# --- File Upload ---
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # --- Preprocessing ---
    st.write("### Data Preprocessing")
    df = df.dropna()

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    # Features & Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Selection ---
    model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Results ---
    st.write("### Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --- Predict on Single Input ---
    st.sidebar.subheader("🔮 Predict Single Customer")
    input_data = []
    for col in df.drop("Churn", axis=1).columns:
        val = st.sidebar.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(val)

    if st.sidebar.button("Predict Churn"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        st.sidebar.success("✅ Customer is likely to Churn" if prediction[0] == 1 else "❌ Customer is NOT likely to Churn")

else:
    st.info("👆 Please upload a CSV file to get started.")
