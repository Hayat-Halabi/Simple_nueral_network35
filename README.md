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
