
# Clinical Stroke Risk Prediction using Machine Learning
![image alt](https://github.com/irfanulkabirhira/explainable-hybrid-rf-et-stroke-risk/blob/f03a61ea1dbbc84f60da38c856e5b5f8eb50a1bf/Screenshot%202026-03-23%20104613.png)

A complete machine learning pipeline for **clinical stroke risk prediction** using data visualization, missing value imputation, class balancing with **SMOTE**, multiple model comparison, and a **hybrid Random Forest + Extra Trees ensemble** with threshold tuning.

---

## Overview

This project aims to predict whether an individual is likely to suffer from a **stroke** based on clinical and demographic features.

The workflow includes:

- Extensive **data visualization**
- Missing value handling using **Decision Tree-based BMI imputation**
- Feature encoding and preprocessing
- Handling data imbalance using **SMOTE**
- Training and evaluating multiple machine learning models
- Building a **hybrid Random Forest + Extra Trees model**
- Performing **decision threshold tuning**
- Interpreting model performance using evaluation metrics and confusion matrices

---

## Dataset

The dataset used in this project is:

**Clinical Stroke Risk Prediction Dataset.csv**

### Features used
- Gender
- Age
- Hypertension
- Heart Disease
- Ever Married
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

### Target variable
- **Stroke**
  - `0` = No Stroke
  - `1` = Stroke

### Dataset shape
- **Rows:** 22,420
- **Columns:** 13

---

## Project Objectives

The main objectives of this project are:

1. Explore the dataset through detailed visualization
2. Identify important factors associated with stroke risk
3. Handle missing BMI values intelligently
4. Solve class imbalance using SMOTE
5. Train multiple machine learning models
6. Select the best-performing model based on **F1-score**
7. Improve prediction performance using a **hybrid ensemble**
8. Tune classification threshold for better precision-recall tradeoff

---

## Workflow

### 1. Data Loading
The CSV file is uploaded and loaded into a pandas DataFrame.

### 2. Missing Value Handling
The `bmi` column contained missing values.

Instead of simple mean or median imputation, a **DecisionTreeRegressor** was used to predict missing BMI values based on:
- Age
- Gender

This provides a more data-driven imputation strategy.

### 3. Exploratory Data Analysis
Several visualizations were created to understand the dataset:

- Distribution of age, glucose level, and BMI
- Stroke vs no-stroke distribution
- Age-wise stroke risk trend
- Gender-wise distribution
- Smoking status comparison
- Heart disease and hypertension trends
- Work type and glucose distribution

### Key insights
- **Age** is one of the strongest indicators of stroke
- Higher **average glucose level** is associated with stroke
- **BMI** shows some differences between stroke and no-stroke groups
- Stroke data is highly **imbalanced**

### 4. Feature Encoding
Categorical features were encoded into numeric form for model training.

### 5. Train-Test Split
The dataset was divided into:
- **Training set**
- **Testing set**

### 6. Class Balancing
Since stroke cases were much fewer than non-stroke cases, **SMOTE** was used on the training data to balance the class distribution.

### 7. Model Training
The following machine learning models were trained and compared:

- Random Forest
- Extra Trees
- CatBoost
- LightGBM
- XGBoost
- MLP
- Gradient Boosting
- SVM
- Logistic Regression
- Naive Bayes

### 8. Cross-Validation
All models were evaluated using **10-fold cross-validation** with **F1-score** as the primary metric.

### 9. Hybrid Model
A hybrid soft-voting ensemble was built using:

- **Random Forest**
- **Extra Trees**

### 10. Threshold Tuning
The final hybrid model’s predicted probabilities were tuned at different thresholds:
- 0.3
- 0.4
- 0.5
- 0.6
- 0.7
- 0.8

This allowed better control over the **precision-recall tradeoff**.

---

## Models Compared

| Model | Mean CV F1 Score |
|------|------:|
| Extra Trees | 0.9830 |
| Random Forest | 0.9827 |
| CatBoost | 0.9453 |
| MLP | 0.9279 |
| XGBoost | 0.9246 |
| LightGBM | 0.9245 |
| Gradient Boosting | 0.8917 |
| SVM | 0.8548 |
| Logistic Regression | 0.8086 |
| Naive Bayes | 0.8023 |

---

## Test Set Results

### Baseline model comparison on test data

| Model | F1 | Accuracy | Recall | Precision | ROC AUC |
|------|------:|------:|------:|------:|------:|
| Random Forest | 0.624 | 0.964 | 0.674 | 0.581 | 0.826 |
| SVM | 0.218 | 0.809 | 0.597 | 0.133 | 0.708 |
| Logistic Regression | 0.209 | 0.776 | 0.664 | 0.124 | 0.723 |
| MLP | 0.305 | 0.883 | 0.574 | 0.208 | 0.736 |
| Naive Bayes | 0.216 | 0.782 | 0.674 | 0.129 | 0.731 |
| Gradient Boosting | 0.265 | 0.842 | 0.639 | 0.167 | 0.745 |
| Extra Trees | 0.660 | 0.968 | 0.697 | 0.626 | 0.839 |
| XGBoost | 0.300 | 0.876 | 0.597 | 0.200 | 0.743 |
| LightGBM | 0.303 | 0.877 | 0.601 | 0.203 | 0.746 |
| CatBoost | 0.371 | 0.905 | 0.630 | 0.263 | 0.774 |

---

## Hybrid Random Forest + Extra Trees Results

### Default threshold = 0.5

- **Accuracy:** 0.9661
- **Recall:** 0.6800
- **Precision:** 0.6071
- **F1 Score:** 0.6415
- **ROC AUC:** 0.8297

### Tuned threshold = 0.8

- **Accuracy:** 0.9792
- **Recall:** 0.6286
- **Precision:** 0.8696
- **F1 Score:** 0.7297
- **ROC AUC:** 0.8121

### Best observation
Threshold tuning significantly improved **precision** and **F1-score**, making the final model more reliable for identifying stroke cases with fewer false positives.

---

## Why F1-Score Was Used

The dataset is highly imbalanced, meaning the number of non-stroke cases is much larger than stroke cases.

In such cases, **accuracy alone can be misleading**.

Therefore, **F1-score** was used as the primary metric because it balances:
- **Precision**
- **Recall**

This makes it more suitable for medical risk prediction tasks.

---

## Why Training Accuracy Became 1.0000

The hybrid model achieved perfect training accuracy because:

- Random Forest and Extra Trees are high-capacity ensemble models
- SMOTE creates a more separable balanced training dataset
- Tree-based models can easily fit resampled training data
- The evaluation was done on the balanced training set after oversampling

So, this is expected and does not necessarily indicate an error.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- CatBoost

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
