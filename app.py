import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="User Conversion Predictor", layout="centered")
st.title("📊 User Conversion Prediction Dashboard")

MODEL_PATH = "model/conversion_model.pkl"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load or train model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.info("ℹ️ Model not found. Training a new model...")

    # Dummy training data (works on Hugging Face)
    X = pd.DataFrame({
        "sessions": [1, 2, 5, 10, 20, 30],
        "pageviews": [5, 10, 25, 50, 80, 120],
        "timeOnSite": [100, 300, 600, 1200, 2400, 3600]
    })
    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    st.success("✅ Model trained and saved successfully!")

# User Inputs
sessions = st.slider("Sessions", 1, 100, 10)
pageviews = st.slider("Pageviews", 1, 200, 20)
timeOnSite = st.slider("Time on Site (seconds)", 10, 5000, 300)

input_data = pd.DataFrame([[sessions, pageviews, timeOnSite]],
                          columns=["sessions", "pageviews", "timeOnSite"])

# Prediction
if st.button("🔮 Predict Conversion"):
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"💡 Conversion Probability: **{prob*100:.2f}%**")