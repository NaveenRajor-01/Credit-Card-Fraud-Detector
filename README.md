# Credit Card Fraud Detection

A Machine Learning based project that detects fraudulent credit card transactions using Logistic Regression.

## Project Overview

This project uses a credit card transaction dataset to classify transactions into two categories:

- 0 → Normal Transaction
- 1 → Fraudulent Transaction

The dataset is highly imbalanced, so under-sampling technique is applied to balance the data before training the model.

## Features

- Load and analyze credit card transaction data
- Check missing values
- Perform data preprocessing
- Handle imbalanced dataset
- Train Machine Learning model
- Predict fraudulent transactions
- Evaluate model accuracy

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Google Colab

## Machine Learning Model

Algorithm Used:

- Logistic Regression

Evaluation Metric:

- Accuracy Score

## Project Workflow

1. Import required libraries
2. Load dataset
3. Explore dataset information
4. Separate legitimate and fraudulent transactions
5. Balance dataset using under-sampling
6. Split data into training and testing sets
7. Train Logistic Regression model
8. Check accuracy of the model

## Files
credit_card_fraud.py
credit_card_fraud_detection.ipynb
creditcard.csv
README.md