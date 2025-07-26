import joblib
from pathlib import Path
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data():
    california = fetch_california_housing()
    return california


def train_model(data):
    X, y = data.data, data.target
    print("Data loaded successfully.")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
   
    model = LinearRegression()
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. Test MSE: {mse:.4f}")
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print("Model saved as linear_regression.joblib")

if __name__ == "__main__":
    california = load_data()
    model = train_model(california)
    model_path = Path("models/linear_regression.joblib")
    save_model(model, model_path)
    print("Training complete.")