"""
Life Expectancy (WHO) - Streamlit App
A simple beginner-friendly app to explore the data and make predictions.
Run this app with:  streamlit run app.py
This app expects the following project structure (run it from the project root):
  project/
  ├── app.py
  ├── data/
  │   └── cleaned_life_expectancy.csv
  └── model/
      └── all the .pkl model files
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Life Expectancy Explorer", layout="wide")

# ---------------------------------------------------------
# Load data (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # read the cleaned, feature-selected dataset saved from the notebook
    data = pd.read_csv("data/cleaned_life_expectancyyy.csv")
    return data

df = load_data()

# ---------------------------------------------------------
# Load saved models (cached so they only load once)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "regression": {
            "Linear Regression": joblib.load("model/reg_linear_regression.pkl"),
            "Decision Tree": joblib.load("model/reg_decision_tree.pkl"),
            "Random Forest": joblib.load("model/reg_random_forest.pkl"),
        },
        "classification": {
            "Logistic Regression": joblib.load("model/clf_logistic_regression.pkl"),
            "Decision Tree": joblib.load("model/clf_decision_tree.pkl"),
            "Random Forest": joblib.load("model/clf_random_forest.pkl"),
            "KNN": joblib.load("model/clf_knn.pkl"),
            "SVM": joblib.load("model/clf_svm.pkl"),
        }
    }
    return models

models = load_models()

# feature columns used by the models (already feature-selected in the notebook)
reg_features = df.drop(columns=["Life expectancy"]).columns.tolist()
clf_features = df.drop(columns=["Status"]).columns.tolist()

# ---------------------------------------------------------
# Sidebar - choose task
# ---------------------------------------------------------
st.sidebar.header("Settings")
task = st.sidebar.radio("Choose a task:", ["Predict Life Expectancy (Regression)", "Predict Country Status (Classification)"])

# ---------------------------------------------------------
# Top row: Title on the left, Predict button on the top right
# (Predict button only shown for the Regression task; Classification updates live)
# ---------------------------------------------------------
title_col, button_col = st.columns([4, 1])
with title_col:
    st.title("🌍 Life Expectancy (WHO) - Model Explorer")
    st.write("A simple app to explore the dataset and test machine learning models.")
with button_col:
    if task == "Predict Life Expectancy (Regression)":
        st.write("")
        st.write("")
        predict_clicked = st.button("🔮 Predict", use_container_width=True)
    else:
        predict_clicked = False

# ===========================================================
# TASK 1: REGRESSION
# ===========================================================
if task == "Predict Life Expectancy (Regression)":

    st.header("📈 Predict Life Expectancy")

    # dropdown to choose the model
    model_name = st.sidebar.selectbox("Choose a regression model:", list(models["regression"].keys()))
    pipe = models["regression"][model_name]

    st.subheader("1. Enter Data Values")
    st.write("Move the sliders to describe a country's health and economic data.")

    # build input sliders for each feature, using dataset min/max/mean as defaults
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(reg_features):
        col = cols[i % 3]
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        if feature == "Status":
            input_data[feature] = col.selectbox("Status (0=Developing, 1=Developed)", [0, 1])
        else:
            input_data[feature] = col.slider(feature, min_val, max_val, mean_val)

    # turn the input into a dataframe with the correct column order
    input_df = pd.DataFrame([input_data])[reg_features]

    st.subheader("2. Prediction")

    # only show a prediction after the Predict button is clicked
    if predict_clicked:
        prediction = pipe.predict(input_df)[0]

        # compute test scores to use as an approximate confidence measure
        X = df.drop(columns=["Life expectancy"])
        y = df["Life expectancy"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        # turn R2 into a percentage to show as a simple "confidence" of the model
        confidence_pct = max(0, r2) * 100

        p1, p2 = st.columns(2)
        p1.metric("Predicted Life Expectancy", f"{prediction:.1f} years")
        p2.metric("Model Confidence (R2)", f"{confidence_pct:.1f}%")

        st.subheader("3. Model Performance (on test data)")
        m1, m2, m3 = st.columns(3)
        m1.metric("R2 Score", round(r2, 3))
        m2.metric("RMSE", round(rmse, 3))
        m3.metric("MAE", round(mae, 3))

        st.subheader("4. Compare All Regression Models")
        compare_rows = []
        for name, m in models["regression"].items():
            p = m.predict(X_test)
            compare_rows.append({
                "Model": name,
                "R2 Score": round(r2_score(y_test, p), 3),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, p)), 3),
                "MAE": round(mean_absolute_error(y_test, p), 3),
            })
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
    else:
        st.info("Set the values above and click the Predict button (top right) to see the result.")

# ===========================================================
# TASK 2: CLASSIFICATION
# ===========================================================
else:
    st.header("🧭 Predict Country Status (Developed vs Developing)")

    # dropdown to choose the model
    model_name = st.sidebar.selectbox("Choose a classification model:", list(models["classification"].keys()))
    pipe = models["classification"][model_name]

    st.subheader("1. Enter Data Values")
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(clf_features):
        col = cols[i % 3]
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_data[feature] = col.slider(feature, min_val, max_val, mean_val)

    input_df = pd.DataFrame([input_data])[clf_features]

    st.subheader("2. Prediction & Confidence")

    # this section updates live as you move the sliders or change the model, no button needed
    prediction = pipe.predict(input_df)[0]
    label = "Developed" if prediction == 1 else "Developing"

    # show probability / confidence as a percentage if the model supports it
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(input_df)[0]
        confidence_pct = max(proba) * 100
        proba_df = pd.DataFrame({
            "Status": ["Developing", "Developed"],
            "Probability (%)": [proba[0] * 100, proba[1] * 100]
        })
    else:
        confidence_pct = None
        proba_df = None

    p1, p2 = st.columns(2)
    p1.metric("Predicted Status", label)
    if confidence_pct is not None:
        p2.metric("Prediction Confidence", f"{confidence_pct:.1f}%")
    else:
        p2.metric("Prediction Confidence", "N/A")

    if proba_df is not None:
        st.write("Probability breakdown:")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x="Status", y="Probability (%)", data=proba_df, ax=ax)
        ax.set_ylim(0, 100)
        st.pyplot(fig)

    st.subheader("3. Model Performance (on test data)")
    X = df.drop(columns=["Status"])
    y = df["Status"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preds = pipe.predict(Xc_test)

    acc = accuracy_score(yc_test, preds)
    prec = precision_score(yc_test, preds)
    rec = recall_score(yc_test, preds)
    f1 = f1_score(yc_test, preds)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Precision", f"{prec*100:.1f}%")
    m3.metric("Recall", f"{rec*100:.1f}%")
    m4.metric("F1 Score", f"{f1*100:.1f}%")

    # confusion matrix
    cm = confusion_matrix(yc_test, preds)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Developing", "Developed"], yticklabels=["Developing", "Developed"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("4. Compare All Classification Models")
    compare_rows = []
    for name, m in models["classification"].items():
        p = m.predict(Xc_test)
        compare_rows.append({
            "Model": name,
            "Accuracy": f"{accuracy_score(yc_test, p)*100:.1f}%",
            "Precision": f"{precision_score(yc_test, p)*100:.1f}%",
            "Recall": f"{recall_score(yc_test, p)*100:.1f}%",
            "F1 Score": f"{f1_score(yc_test, p)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

    # -------------------------------------------------
    # Decision Boundary (using only 2 features, for visualization only)
    # Updates live as soon as you change the dropdowns, no button needed
    # -------------------------------------------------
    st.subheader("5. Decision Boundary (2D View)")
    st.write("This plot trains a quick model on just 2 features so we can draw a 2D boundary. "
             "It is only for visualization and is separate from the main model above.")

    b1, b2 = st.columns(2)
    feature_x = b1.selectbox("Feature for X-axis:", clf_features, index=clf_features.index("Schooling"))
    feature_y = b2.selectbox("Feature for Y-axis:", clf_features, index=clf_features.index("Adult Mortality"))

    # cache this so picking a new feature pair or model doesn't retrain from scratch every time
    @st.cache_resource
    def train_boundary_model(model_name, feature_x, feature_y):
        # pick the same type of model the user chose, for the 2D boundary
        boundary_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
        }
        # models that need scaling for the 2D boundary too
        boundary_needs_scaling = {"Logistic Regression", "KNN", "SVM"}

        # build a small pipeline using only the two chosen features
        X2 = df[[feature_x, feature_y]]
        y2 = df["Status"]
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

        if model_name in boundary_needs_scaling:
            b_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", boundary_models[model_name])
            ])
        else:
            b_pipe = Pipeline([
                ("model", boundary_models[model_name])
            ])
        b_pipe.fit(X2_train, y2_train)
        return b_pipe

    boundary_pipe = train_boundary_model(model_name, feature_x, feature_y)

    # create a grid of points to plot the decision boundary
    x_min, x_max = df[feature_x].min() - 1, df[feature_x].max() + 1
    y_min, y_max = df[feature_y].min() - 1, df[feature_y].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})
    zz = boundary_pipe.predict(grid_points).reshape(xx.shape)

    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    ax_b.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    sns.scatterplot(x=feature_x, y=feature_y, hue=df["Status"], data=df, ax=ax_b, palette="coolwarm", edgecolor="k", alpha=0.7)
    ax_b.set_title(f"Decision Boundary: {model_name} ({feature_x} vs {feature_y})")
    st.pyplot(fig_b)

st.sidebar.markdown("---")
st.sidebar.caption("Built for the JIIT Summer Internship Report | Life Expectancy (WHO) Dataset")