import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Set random seed for reproducibility of synthetic data
np.random.seed(42)

# Generate synthetic drone data: 100 samples with 2 features [battery_level (%), altitude (meters)]
# Battery level ranges from 0-100%, altitude from 0-500 meters
X = np.random.rand(100, 2) * [100, 500]

# Create labels: 1 (land) if battery < 20% or altitude < 50m, else 0 (continue flying)
y = np.where((X[:, 0] < 20) | (X[:, 1] < 50), 1, 0)

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier with max depth of 3 to prevent overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the model's accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Calculate and print the confusion matrix to show true/false positives/negatives
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"[[True Negatives (Continue Flying)  False Positives (Predicted Land)]")
print(f" [False Negatives (Predicted Fly)  True Positives (Land)]]")
print(cm)

# Example prediction for a new drone state: [battery_level=15%, altitude=30m]
new_data = np.array([[15, 30]])

# Predict whether the drone should land or continue flying
prediction = model.predict(new_data)
print(f"Prediction for battery 15%, altitude 30m: {'Land' if prediction[0] == 1 else 'Continue Flying'}")
