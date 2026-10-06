# Titanic Survival Prediction using Machine Learning

## Introduction

This project demonstrates how to optimize a machine learning pipeline using the **Titanic Survival Dataset**. The goal is to build a classification model that predicts whether a passenger survived the sinking of the Titanic based on passenger attributes such as age, sex, class, and fare.

The project begins by building a **Random Forest Classifier** and then updates the pipeline to use a **Logistic Regression** classifier. Both models are evaluated and compared using cross-validation and hyperparameter tuning to determine which performs better.

This project provides hands-on experience with building, optimizing, and evaluating machine learning pipelines using **scikit-learn**.

---

## Project Objectives

- Build a machine learning model to solve a classification problem using **scikit-learn**.
- Create a preprocessing and modeling pipeline using **Pipeline**.
- Apply **cross-validation** to evaluate model performance.
- Perform **hyperparameter tuning** using **GridSearchCV**.
- Interpret classification model results using common evaluation metrics.
- Replace one classifier with another while keeping the same preprocessing pipeline.
- Compare the performance of multiple classification algorithms.

---

## Dataset

The project uses the **Titanic Survival Dataset**, which contains information about passengers aboard the Titanic.

The target variable is:

- **Survived**
  - `1` = Survived
  - `0` = Did Not Survive

Example features include:

- Passenger Class (Pclass)
- Sex
- Age
- Fare
- Number of Siblings/Spouses Aboard (SibSp)
- Number of Parents/Children Aboard (Parch)
- Embarked Port

---

## Machine Learning Models

This project compares two classification algorithms:

- Random Forest Classifier
- Logistic Regression

---

## Workflow

1. Load and explore the dataset.
2. Clean and preprocess the data.
3. Split the dataset into training and testing sets.
4. Build a preprocessing pipeline.
5. Train a Random Forest classifier.
6. Optimize the model using GridSearchCV.
7. Replace the classifier with Logistic Regression.
8. Tune hyperparameters.
9. Evaluate both models using cross-validation and test data.
10. Compare model performance and discuss the results.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Evaluation Metrics

The models are evaluated using several classification metrics, including:

- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation Score
- Confusion Matrix

---

## Learning Outcomes

This project demonstrates how to:

- Build reusable machine learning pipelines.
- Combine preprocessing and model training into a single workflow.
- Tune hyperparameters efficiently using GridSearchCV.
- Compare multiple classification models objectively.
- Apply best practices for supervised machine learning using scikit-learn.
