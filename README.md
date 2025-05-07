# COVID-19 Hospitalization Prediction

This project predicts weekly changes in COVID-19 hospitalization rates across various countries using machine learning techniques. The project involved both classification and regression tasks, using real-world COVID-19 data, rigorous preprocessing, feature engineering, and model selection strategies.

## 📍 Project Link
The core notebook is available on Google Colab:  
👉 [Open in Colab](https://colab.research.google.com/drive/1SEyIAVMZ-5tp2nvYnz17FY_LVd7iNQdM#scrollTo=ubTvGmn7ZUTA)

## 📊 Problem Formulation & Preprocessing

We framed the task as both a binary classification problem (predicting increase/no increase in hospitalizations) and a regression problem (predicting exact values of next_week_hospitalizations). The dataset was standardized **per country**, using the mean and standard deviation of each feature within each country to account for population/resource disparities. Country names were one-hot encoded, and non-informative fields like raw dates and weeks were dropped.

### Key preprocessing steps:
- Standardization per country
- One-hot encoding of country names
- Feature pruning (e.g., removing low-impact age groups like under-15, 15–24)
- Dropping irrelevant temporal features like `date` and `week` for simplification

## 🧠 Models Used

### Baseline Classification
- **Support Vector Machine (SVM)** with RBF kernel
- **Random Forest Classifier**
- Hyperparameter tuning done via grid search on `C` and `gamma` using Stratified Shuffle Split cross-validation
- Achieved **79% classification accuracy** after feature pruning

### Regression for Final Task
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Linear, Ridge, and Lasso Regression**
- **SVR and MLP Regressor (exploratory, but underperformed)**

Special modeling approaches:
- Ensemble methods trained per country to leverage country-specific trends
- GridSearch and manual tuning of hyperparameters (e.g., `alpha`, `C`)
- Feature testing with temporal variables (`season`, `date`) that were later dropped to improve performance

## ✅ Results

| Model | Task | Metric | Result |
|-------|------|--------|--------|
| SVM | Classification | Accuracy | 79% |
| Ridge Regression | Regression | MSE | **147k** |
| Gradient Boosting | Regression | MSE | 450k (Overfit) |
| Linear Regression | Regression | MSE | 131k (Best) |

We reached and surpassed the 150k MSE threshold using linear and ridge regression after careful preprocessing and per-country modeling.

## 📂 Repository Structure

```
COVID19-Hospitalization-Prediction/
│
├── notebooks/
│   └── covid_hospitalization_prediction.ipynb
├── data/
│   └── (Instructions to download or placeholder for data)
├── results/
│   └── kaggle_submission_screenshot.png
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## 📦 Requirements

See `requirements.txt` for dependencies. You can install them using:

```bash
pip install -r requirements.txt
```

## 📌 Authors

Project by Mohamed Abdalla and team  
Cornell University – COVID-19 Hospitalization ML Project (Spring 2021)
