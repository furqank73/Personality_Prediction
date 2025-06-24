# 🧠 Personality Prediction Web App

A professional and interactive **Streamlit web application** that predicts whether a person is **Introverted** or **Extroverted** based on behavioral traits. The project combines machine learning with insightful visualizations and a clean UI to deliver meaningful personality predictions.

---

## 🚀 Demo

🔗 **Live App**: *Coming Soon*  
📽️ **Demo Video**: *Coming Soon*

---

## 📌 Features

- 🧾 Upload dataset or enter individual traits for prediction
- 🧍‍♂️ Predict personality (Introvert/Extrovert) using behavioral inputs
- 📊 Personality Spectrum Gauge (Introversion Level)
- 📈 Behavioral Comparison Table for feature interpretation
- 🧠 Interactive and visually engaging charts (Plotly)
- 🎨 Clean and modern Streamlit UI with custom CSS

---

## 🧠 Model Development Workflow

The model was developed through a structured and explainable machine learning workflow as documented in the [Jupyter Notebook](https://github.com/furqank73/Personality_Prediction/personality.ipynb):

### 1. 📥 Data Loading & Exploration

- Loaded behavioral data with features like time spent alone, stage fear, etc.
- Explored value distributions and class balance using `pandas` and `seaborn`.

### 2. 📊 Exploratory Data Analysis (EDA)

- Visualized behavioral patterns using boxplots, countplots, and heatmaps.
- Key differences between introverts and extroverts were identified.

### 3. 🧹 Data Preprocessing

- Filled missing values (e.g. median imputation).
- Cleaned and standardized column names.
- Converted binary categorical values (Yes/No) to numeric (0/1).

### 4. 🧪 Feature Selection & Engineering

Final input features used:
- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`

Target: `Personality` (0 = Introvert, 1 = Extrovert)

### 5. 📈 Model Building & Training

- Performed an 80/20 train-test split
- Trained multiple models:
  - Logistic Regression
  - Random Forest (Best)
  - KNN
- Selected the best model based on evaluation metrics.

### 6. ✅ Model Evaluation

- Accuracy, Precision, Recall, and F1-score calculated
- Confusion matrix visualized
- Random Forest provided the most balanced performance

### 7. 💾 Model Export

- Final model exported using:
  ```python
  joblib.dump(model, 'model/personality_model.pkl')
