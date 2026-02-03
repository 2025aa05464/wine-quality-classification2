import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Wine Quality Classification 🍷")

# Upload CSV
uploaded_file = st.file_uploader("Upload Wine Quality CSV", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file, sep=";")
    st.write("Dataset Preview:", df.head())

    # Features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Model selection dropdown
    model_choice = st.selectbox(
        "Choose a model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    # Load pipeline model
    model_path = f"model/{model_choice.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X)

    # Results
    st.subheader("Predictions")
    st.write(y_pred)

    # Evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"MCC: {matthews_corrcoef(y, y_pred):.2f}")

    # Confusion Matrix Heatmap
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)