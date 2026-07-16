# 🌍 Life Expectancy (WHO) — Data Analysis & Prediction App

A beginner-friendly machine learning project that analyzes the WHO Life Expectancy dataset, builds simple prediction models, and serves them through an interactive Streamlit app.

Built as part of a B.Tech Summer Internship(JIIT Noida).

---

## 📊 Dataset

- **Source:** [Kaggle — Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Rows:** ~2,900 records across countries and years
- **Target variables:**
  - `Life expectancy` — continuous value (regression task)
  - `Status` — Developed / Developing (classification task)

---

## 🧠 What This Project Does

1. **Cleans** the raw data (fixes column names, fills missing values with the median)
2. **Explores** the data with plots (histograms, boxplots, correlation heatmap, scatter plots)
3. **Selects relevant features** — keeps only the columns most correlated with life expectancy, drops weak/redundant ones (Population, Measles, Year, etc.)
4. **Scales features selectively** — only models that need it (Linear/Logistic Regression, KNN, SVM) get `StandardScaler`; tree-based models (Decision Tree, Random Forest) train on raw values
5. **Trains multiple models** for both tasks and compares them with proper metrics
6. **Serves everything** through an interactive Streamlit app with live predictions, confidence scores, and a decision boundary visualizer

---

## 🗂️ Project Structure

```
Life-Expectancy-Prediction & classification/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── Life Expectancy Data.csv
│   └── cleaned_life_expectancy.csv
│
├── model/
│   ├── reg_linear_regression.pkl
│   ├── reg_decision_tree.pkl
│   ├── reg_random_forest.pkl
│   ├── clf_logistic_regression.pkl
│   ├── clf_decision_tree.pkl
│   ├── clf_random_forest.pkl
│   ├── clf_knn.pkl
│   └── clf_svm.pkl
│
├── code/
│   └── life_expectancy_analysis.ipynb

```

---

## ⚙️ Models Used

| Task | Models | Scaling Applied |
|---|---|---|
| Regression (Life Expectancy) | Linear Regression, Decision Tree, Random Forest | Linear Regression only |
| Classification (Status) | Logistic Regression, Decision Tree, Random Forest, KNN, SVM | Logistic Regression, KNN, SVM |

**Evaluation metrics:**
- Regression → R² Score, RMSE, MAE
- Classification → Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/aadityapal183/Life_Expectancy_Prediction_and_classification.git
cd Life_Expectancy_Prediction_and_classification
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
```
or
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `life_expectancy_analysis.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run all cells top to bottom. This regenerates the cleaned dataset and all `.pkl` model files.

## 4. Run the Streamlit App

Make sure the project structure is set up correctly and all dependencies are installed.

Run the app using either of the following commands:

```bash
py -m streamlit run app.py
```

**OR**

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```
http://localhost:8501
```

## 🖥️ App Features

- **Task switcher** — toggle between predicting Life Expectancy (regression) and Country Status (classification)
- **Model dropdown** — pick from multiple trained models per task
- **Predict button** — top-right; nothing runs until you click it
- **Confidence scores** — shown as a percentage (R²-based for regression, `predict_proba`-based for classification)
- **Model comparison table** — see all models' metrics side by side
- **Decision boundary plot** — pick any two features and visualize how a classifier separates Developed vs Developing countries

---

## 📚 Learning Outcomes

- Real-world data cleaning and handling missing values
- Feature selection using correlation analysis
- Understanding when feature scaling is necessary vs. unnecessary
- Building ML pipelines with `scikit-learn`
- Comparing regression and classification models using proper metrics
- Deploying a trained model with a simple interactive web app (Streamlit)

---

## 🔮 Future Scope

- Add deep learning models for improved accuracy
- Deploy the app on Streamlit Community Cloud or similar hosting
- Incorporate more recent WHO datasets
- Add a mobile-friendly version of the app

---

## 📄 License

This project is for academic/educational purposes as part of a summer internship report.
