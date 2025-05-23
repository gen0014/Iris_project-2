# 🌸 Iris Flower Classification with AI Explainability

This project demonstrates a machine learning and AI-powered system for classifying Iris flowers into species based on flower measurements. It includes a web app built with Streamlit and uses SHAP (SHapley Additive exPlanations) for AI interpretability.

## 📦 Features

- Classifies Iris flowers into Setosa, Versicolor, or Virginica
- Built with Random Forest Classifier
- Interactive Streamlit web app interface
- SHAP-based AI explainability (global + local)
- Documentation and presentation included

## 📊 Dataset

- Dataset: Iris dataset from `sklearn.datasets`
- Features: Sepal length, Sepal width, Petal length, Petal width
- Target: Species (Setosa, Versicolor, Virginica)

## 🧠 AI Interpretability (SHAP)

- SHAP provides a way to understand how each feature contributes to model predictions.
- The app includes:
  - Summary plots for global feature importance
  - Force plots (explanation of individual predictions)

## 🚀 How to Run the App

1. Install the requirements:

```bash
pip install streamlit shap matplotlib scikit-learn pandas

streamlit run app.py
