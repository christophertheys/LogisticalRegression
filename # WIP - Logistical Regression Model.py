# Lab 2 - Logistical Regression Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('/Users/christophertheys/Desktop/diabetes_prediction_dataset.csv')

# Remove Rows with Missing Values
data.dropna(inplace=True)

# Convert gender to numerical values
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

# Apply one-hot encoding to the smoking_history column
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

# Now, we need to update the features list to reflect the new one-hot encoded columns
features = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'] 
features += [col for col in data.columns if 'smoking_history_' in col]

X = data[features]
y = data['diabetes']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

missing_values = data.isnull().sum().sum()
if missing_values > 0:
    print(f"Warning: There are {missing_values} missing values in the dataset.")

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.bar(features, model.coef_[0])
plt.xticks(rotation=45)
plt.title("Coefficients of Logistic Regression Model")
plt.ylabel("Coefficient Value")
plt.show()
