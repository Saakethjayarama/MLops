import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset from a local CSV file
iris = pd.read_csv("dataset/iris.csv")
X = iris.iloc[:, :-1].values  # Features (all columns except the last)
y = iris.iloc[:, -1].values   # Target (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and their parameters
models = [
    {"model": LogisticRegression, "params": {"solver": "liblinear", "multi_class": "auto"}},
    {"model": RandomForestClassifier, "params": {"n_estimators": 100, "random_state": 42}},
    {"model": SVC, "params": {"kernel": "linear", "C": 1.0}},
]

# Start MLflow experiment
mlflow.set_experiment("Iris Model Training")

for model_config in models:
    model_class = model_config["model"]
    params = model_config["params"]

    with mlflow.start_run():
        # Instantiate and train the model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Model: {model_class.__name__}, Accuracy: {accuracy:.4f}")

print("Training complete. Check MLflow UI for details.")
