import pandas as pd
df = pd.read_csv(r'C:\Users\Shweta Jha\OneDrive\Desktop\MY projects\diabetes.csv')
print(df.shape)
df.head()
df.info()
import seaborn as sns
import matplotlib.pyplot as plt

# Class distribution
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Replace 0s with median for relevant features
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
import numpy as np

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.show()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\Shweta Jha\OneDrive\Desktop\MY projects\diabetes.csv")


# Replace 0s in key features with median values
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

# Split into X, y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split & train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient details to estimate diabetes risk.")

# Input sliders
def user_input():
    Pregnancies = st.slider("Pregnancies", 0, 17, 1)
    Glucose = st.slider("Glucose", 50, 200, 100)
    BloodPressure = st.slider("Blood Pressure", 30, 122, 70)
    SkinThickness = st.slider("Skin Thickness", 0, 99, 20)
    Insulin = st.slider("Insulin", 0, 846, 79)
    BMI = st.slider("BMI", 10.0, 67.0, 30.0)
    DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.slider("Age", 15, 100, 33)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input()

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
pred_prob = model.predict_proba(input_scaled)[0][1]

st.subheader("🔍 Prediction")
st.write("**Diabetes Risk:**", "Positive 🔴" if prediction[0] == 1 else "Negative 🟢")
st.write("**Risk Probability:**", f"{pred_prob:.2%}")
