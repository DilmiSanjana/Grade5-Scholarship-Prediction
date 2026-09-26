import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset/students.csv")

# Features and target
X = data.drop("result", axis=1)
y = data["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "model/scholarship_model.pkl")

print("Model saved successfully!")