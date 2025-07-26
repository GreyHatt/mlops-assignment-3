import joblib
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing

class SingleLayerPyTorch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
   
    def forward(self, x):
        return self.linear(x)

def load_data():
    california = fetch_california_housing()
    return california

def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def run_model(model, data):
    X_test = data.data
    input_dim = X_test.shape[1]
    pytorch_model = SingleLayerPyTorch(input_dim)
   
    with torch.no_grad():
        pytorch_model.linear.weight.data = torch.FloatTensor(model.coef_.reshape(1, -1))
        pytorch_model.linear.bias.data = torch.FloatTensor([model.intercept_])
   
    X_test_tensor = torch.FloatTensor(X_test)
    pytorch_model.eval()
    predictions = pytorch_model(X_test_tensor)
   
    print("Sample predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"Sample {i+1}: {pred.item():.2f}")

if __name__ == "__main__":
    california = load_data()
    model_path = Path("models/linear_regression.joblib")
    model = load_model(model_path)
    run_model(model, california)