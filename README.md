# Diabetes-Analysis
A Streamlit-powered web app that predicts the likelihood of diabetes onset using diagnostic health data.

## Project Structure

diabetes-prediction/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
│
├── diabetes-risk-app/
├── app.py
├── data/
│   └── diabetes.csv
├── requirements.txt
├── README.md


##  Key Features:
User-friendly web interface for inputting patient data

Trained Random Forest model on Pima Indians Diabetes dataset

Real-time prediction with risk probability

Clean visual layout, easy to use

## Dataset:
Pima Indians Diabetes Database

Includes features like Glucose, BMI, Age, Insulin, etc.

## Tech Stack:
Python · Streamlit · scikit-learn · pandas · matplotlib

## How to Run:
pip install -r requirements.txt
streamlit run app.py
