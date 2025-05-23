# app.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Classification with AI Explainability")
st.write("Enter flower measurements below to predict the species and explain the decision:")

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Display prediction
st.subheader("Prediction:")
st.write(f"**Predicted Species:** {iris.target_names[prediction]}")
st.write("**Prediction Probabilities:**")
st.bar_chart(prediction_proba[0])

# SHAP Explainability
st.subheader("AI Explanation with SHAP")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Use matplotlib to generate SHAP force plot
shap.initjs()
shap_input = pd.DataFrame(input_data, columns=iris.feature_names)
st.write("Explanation of model's decision:")

# SHAP summary plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)

# Individual force plot
st.write("Force plot (only viewable in notebook or Jupyter environment):")
st.warning("Note: Force plots are interactive and may not render in Streamlit. For full interactivity, use Jupyter Notebook.")
