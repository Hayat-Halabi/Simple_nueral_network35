# Simple_nueral_network35
In this project, we aim to build a simple neural network to predict the likelihood of an individual having diabetes based on a synthetic dataset of health-related features.

# Diabetes Prediction Using a Simple Neural Network
# Scenario:
Diabetes is a prevalent and serious health condition that affects millions of people worldwide. Early detection and accurate prediction of diabetes are crucial for effective disease management. In this project, we aim to build a simple neural network to predict the likelihood of an individual having diabetes based on a synthetic dataset of health-related features.

# Problem Statement:
You have a synthetic dataset containing information about individuals, including features like age, BMI, blood pressure, and more. The target variable is 'Outcome', which indicates whether the individual has diabetes (1) or not (0). You have to develop a basic neural network from scratch to perform binary classification.

# Directions:
- Ensure you have a CSV file named synthetic_diabetes_dataset.csv in the same directory as your Python script, containing your dataset.

- Import necessary libraries: Pandas, NumPy, and scikit-learn.

- Load the dataset.

- Split data into features (X) and target (y).

- Further split the X and y into training and testing sets (80% training, 20% testing) using train_test_split.

- Initialize parameters (weights and biases) using initialize_parameters.

- Use the sigmoid activation function as sigmoid(z).

- Calculate predictions based on the parameters and input data using the predict function.

- Add training logic.

- Update parameters iteratively to minimize the chosen loss function.

- After training (or using pre-trained parameters), predict on the test set X_test using the predict function.

- Calculate accuracy with accuracy_score by comparing y_test and y_pred.

- Generate a classification report using classification_report for precision, recall, F1-score, and support metrics.

- By following these concise steps, you will execute the code and obtain the output, including model accuracy and a classification report for your diabetes prediction model.

``` python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the synthetic dataset
data = pd.read_csv('synthetic_diabetes_dataset.csv')

# Split the data into features and target variable
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize parameters
def initialize_parameters(n_input, n_hidden, n_output):
    np.random.seed(0)
    W1 = np.random.randn(n_hidden, n_input)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden)
    b2 = np.zeros((n_output, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# Define sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Predict function
def predict(X, parameters):
    # Retrieve parameters
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']

    # Forward propagation
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    predictions = (A2 > 0.5).astype(int)
    return predictions.ravel()

# Initialize parameters
parameters = initialize_parameters(X_train.shape[1], 10, 1)

# Predict on the test set
y_pred = predict(X_test, parameters)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(report)
```
Link to access CSV
https://drive.google.com/file/d/1dYAGGOWTx3_KFu-wVFUGgvWhmHquJXtr/view?usp=drive_link
