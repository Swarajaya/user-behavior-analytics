import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

X_demo = np.array([[0,0],[0,1],[1,0],[1,1]])
y_demo = np.array([0,0,0,1])
model = LogisticRegression()
model.fit(X_demo, y_demo)


st.title("User Conversion Prediction Demo")

st.write("Enter features to predict conversion:")

feature1 = st.number_input("Feature 1", value=0)
feature2 = st.number_input("Feature 2", value=0)

if st.button("Predict Conversion"):
    prediction = model.predict([[feature1, feature2]])
    st.success(f"Predicted Conversion: {prediction[0]}")
