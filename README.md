# Job Salary Prediction 
### Machine Learning-Based Salary Estimation System

<div align="center">
  <img src="https://raw.githubusercontent.com/Haid3rH/job-salary-prediction/main/banner.png" alt="Job Salary Predictor" width="95%">
</div>

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Project Overview

The **Job Salary Prediction** project is a machine learning application designed to **estimate annual salaries** based on demographic and professional features.  

The model is trained on comprehensive salary data and includes **advanced preprocessing techniques**, combining numerical scaling, ordinal encoding, and one-hot encoding to maximize prediction accuracy.  

This project is ideal for:  

- Job seekers wanting to estimate fair market salaries,  
- Students and developers learning regression and preprocessing pipelines,  
- Professionals exploring real-world applications of ML pipelines and Streamlit deployment.  

---

## Why This Project Matters

Salary expectations are often unclear, varying widely depending on education, experience, job role, location, and other factors. Most job seekers struggle to estimate fair compensation or understand the factors driving salary differences.  

This project addresses that gap by providing a **fast, accurate, and interactive ML-based estimator**, helping users make informed decisions about career planning and salary negotiations.  

**Key benefits**:  
- Accurate salary estimation based on multiple professional factors,  
- Interactive exploration of how different features impact compensation,  
- Hands-on example of **preprocessing pipelines and ensemble models**,  
- Useful for both educational purposes and practical salary planning.  

**Who benefits**:  
- Job seekers estimating fair market compensation,  
- Students learning regression models, pipelines, and feature engineering,  
- Developers building predictive analytics apps,  
- HR professionals and recruiters exploring data-driven insights.  

---

## Live Web App

<div style="padding:10px; font-size:100%; text-align:left;">
    URL: 
    <a href="https://job-salary-prediction.streamlit.app/" target="_blank">
        Click here to open the Job Salary Predictor
    </a>
</div>

---

## Dataset Overview — Salary Prediction Dataset

This project uses a comprehensive salary dataset that includes demographic and professional metrics relevant to compensation.  

### **Entries**
- Multiple individual records across various industries

### **Columns**

| Column       | Description                                   |
|--------------|-----------------------------------------------|
| Age          | Age of the individual                         |
| Experience   | Years of professional experience              |
| Education    | Education level (High School to PhD)          |
| Job_Title    | Professional role (Analyst, Engineer, etc.)   |
| Location     | Geographic region (Urban, Suburban, Rural)    |
| Gender       | Gender of the individual                      |
| Salary       | Annual salary in USD (target variable)        |

### **Dataset Details**
- Format: CSV  
- License: Public dataset

---

## Repository Structure
```text
.
Job_Salary_Prediction/
├── profile_img.png                 ← Project banner
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── salary_prediction_data.csv    ← Raw Dataset
│   └── cleaned/
│       └── salary_data_cleaned.csv       ← Cleaned Dataset
|   └── data_report/
|       └── salary.html      ← HTML report of our data 
├── src/
│   └── model_training.ipynb              ← Main training notebook (EDA & modeling)
├── model/
│   └── best_salary_model.pkl             ← Gradient Boosting pkl file
└── app/
    └── app.py                            ← Interactive Streamlit web app
````

---

## How Features Are Processed

1. **Data Cleaning & Preprocessing**

   * Remove duplicates, handle missing values
   * Train-test split (80-20) for robust evaluation

2. **Feature Engineering**

   * Numerical features: Age, Experience (StandardScaler)
   * Ordinal features: Education (High School → Bachelor → Master → PhD), Job_Title (Analyst → Engineer → Manager → Director)
   * Nominal features: Location, Gender (OneHotEncoder)

3. **Encoding & Transformation**

   * StandardScaler for numerical features (Age, Experience)
   * OrdinalEncoder for hierarchical categories (Education, Job_Title)
   * OneHotEncoder for nominal categories (Location, Gender)

4. **Pipeline Construction**

   * Combined preprocessing + model in a `Pipeline` for reproducibility and deployment

5. **Model Selection & Training**

   * Tested multiple regressors: Linear, Ridge, Lasso, ElasticNet, KNN, SVR, DecisionTree, RandomForest, GradientBoosting, AdaBoost, XGBoost
   * Performance on Test Set :
     | Model                  | Test R²  | Test MAE  | Test RMSE |
     |------------------------|----------|-----------|-----------|
     | Gradient Boosting      | 0.8549   | 8,807.56  | 10,885.50 |
     | Random Forest          | 0.8428   | 9,211.60  | 11,328.09 |
     | XGBoost                | 0.8293   | 9,530.39  | 11,807.64 |
     | Decision Tree          | 0.8043   | 10,310.48 | 12,642.24 |
     | ElasticNet             | 0.7748   | 11,284.23 | 13,561.50 |
     | Ridge Regression (L2)  | 0.7745   | 11,291.94 | 13,569.25 |
     | Linear Regression      | 0.7744   | 11,294.55 | 13,572.00 |
     | Lasso Regression (L1)  | 0.7744   | 11,294.55 | 13,572.00 |
     | AdaBoost               | 0.7655   | 11,133.45 | 13,836.19 |
     | KNN Regressor          | 0.7472   | 11,472.56 | 14,368.60 |
     | SVR (RBF)              | 0.0202   | 22,769.50 | 28,285.10 |

   * Gradient Boosting was selected as the final model based on highest R² score and lowest MAE/RMSE,
     making it the most accurate and robust model for predicting job salaries.
   


---

## Run Locally

```bash
git clone https://github.com/Haid3rH/job-salary-prediction.git
cd job-salary-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Future Improvements

* Integrate SHAP/LIME for feature importance visualization
* Add more advanced ML models (e.g., LightGBM or CatBoost)
* Include real-time salary comparison based on industry and company size
* Add historical salary trend analysis

---


---

* ```
            Developed by Haris • Haider 
                   February 2026
      Stars & feedback are highly appreciated
               github.com/Haid3rH
  ```
