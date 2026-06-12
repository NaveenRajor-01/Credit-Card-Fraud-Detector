# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset to a pandas dataframe
credit_card_data = pd.read_csv('./creditcard.csv')

# First 5 rows of the dataset
print(credit_card_data.head())

# Last 5 rows of the dataset
print(credit_card_data.tail())

# Dataset informations
credit_card_data.info()

# Checking the number of missing values in each column
print(credit_card_data.isnull().sum())

# The distribution of the legit transaction and fraudulent transaction
print(credit_card_data['Class'].value_counts())

# This dataset is highly unbalanced
# 0 --> normal transaction
# 1 --> fraud transaction

# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# Statistical measures of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())

# Compare the values for both transactions
print(credit_card_data.groupby('Class').mean())

# Under sample
# Build a sample dataset containing similar distribution of normal and fraudulent transactions
# Number of fraudulent transactions --> 492
legit_sample = legit.sample(n=492)

# Concatenating two dataframes
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

# Splitting the data into features and the targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)

# Split the data into training data and the testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# MODEL TRAINING
# Logistic Regression
model = LogisticRegression()

# Training the logistic regression model with training data
model.fit(X_train, Y_train)

# Model evaluation
# Accuracy score
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on test data : ', test_data_accuracy)
