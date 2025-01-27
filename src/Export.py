import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

# Load the Iris dataset from a local CSV file
iris = pd.read_csv("dataset/iris.csv")
X = iris.iloc[:, :-1].values  # Features (all columns except the last)
y = iris.iloc[:, -1].values   # Target (last column)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and parameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Start MLflow experiment
mlflow.set_experiment("Iris_RandomForest_Regression")

with mlflow.start_run():
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log model parameters, metrics, and artifacts
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", -grid_search.best_score_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("test_mse", test_mse)

    # Log model
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    print("Best Parameters:", best_params)
    print("Test MSE:", test_mse)

# Save the best model to a file for deployment
model_path = "best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Model saved to {model_path}")