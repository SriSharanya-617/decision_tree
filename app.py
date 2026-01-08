import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Custom CSS
# -------------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Decision Tree - Bank Deposit Prediction", layout="wide")

st.title("🏦 Term Deposit Subscription Prediction")
st.write("Decision Tree based Machine Learning Web Application")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Feature & Target
# -------------------------------
X = df.drop("deposit", axis=1)
y = df["deposit"]

X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Decision Tree
# -------------------------------
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
dt_model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = dt_model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
st.subheader("📈 Model Evaluation")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# Decision Tree Visualization
# -------------------------------
st.subheader("🌳 Decision Tree Visualization")

fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    ax=ax2
)
st.pyplot(fig2)

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance)

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🔮 Predict New Customer")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = dt_model.predict(input_df)[0]
    probability = dt_model.predict_proba(input_df)

    st.success(f"Prediction: {prediction}")
    st.write("Probability:", probability)
