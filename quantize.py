import joblib
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

class SingleLayerPyTorch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
   
    def forward(self, x):
        return self.linear(x)

def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def extract_parameters(model):
    weights = np.asarray(model.coef_)
    bias = np.asarray(model.intercept_)
    return weights, bias

def quantize_parameters(params, min_val, max_val, bits=8):
    if max_val == min_val:
        quantized = np.full_like(params, (2**bits - 1) // 2, dtype=np.uint8)
        scale = 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = int(np.round(-min_val / scale))
        quantized = np.clip(
            np.round((params - min_val) / scale) + zero_point,
            0,
            2**bits - 1
        ).astype(np.uint8)

    return quantized, scale, zero_point


def dequantize(quantized, scale, zero_point):
    return scale * (quantized.astype(np.float32) - zero_point)

def run(model):
    weights, bias = extract_parameters(model)
    unquant_params = {
        'weights': weights,
        'bias': bias
    }
    joblib.dump(unquant_params, 'models/unquant_params.joblib')
   
    w_min, w_max = weights.min(), weights.max()
    b_min, b_max = bias.min(), bias.max()
   
    quant_weights, w_scale, w_zp = quantize_parameters(weights, w_min, w_max)
    quant_bias, b_scale, b_zp = quantize_parameters(bias, b_min, b_max)
   
    dequant_weights = dequantize(quant_weights, w_scale, w_zp)
    dequant_bias = dequantize(quant_bias, b_scale, b_zp)

    input_dim = weights.shape[0]
    model_torch = SingleLayerPyTorch(input_dim)
    model_torch.linear.weight.data = torch.from_numpy(dequant_weights.reshape(1, -1).astype(np.float32))
    model_torch.linear.bias.data = torch.tensor([dequant_bias], dtype=torch.float32)


    quant_params = {
        'quant_weights': quant_weights,
        'quant_bias': quant_bias,
        'w_scale': w_scale,
        'w_zp': w_zp,
        'b_scale': b_scale,
        'b_zp': b_zp,
        "state_dict": model_torch.state_dict()
    }
    joblib.dump(quant_params, 'models/quant_params.joblib')
   
    print("Original weights shape:", weights.shape)
    print("Quantization error (weights):", np.mean(np.abs(weights - dequant_weights)))
    print("Original bias shape:", bias.shape)
    print("Quantization error (bias):", np.mean(np.abs(bias - dequant_bias)))
    return model_torch


def quantize_model(inp, n_bits=8):
    levels = 2 ** n_bits -1
    
    val =  np.round(inp * levels).astype(np.uint8)
    return val

def dequantize_model(inp, n_bits=8):
    levels = 2 ** n_bits -1
    
    val =  (inp.astype(np.float64) / levels)
    return val

def fix_run(model):
    weights, bias = extract_parameters(model)
    unquant_params = {
        'weights': weights,
        'bias': bias
    }
    joblib.dump(unquant_params, 'models/unquant_params.joblib')

    w_min = weights.min()
    w_max = weights.max()
    b_min = bias.min()
    b_max = bias.max()
   
    weights_norm = (weights - w_min) / (w_max - w_min) if w_max != w_min else np.zeros_like(weights)
    bias_norm = (bias - b_min) / (b_max - b_min) if b_max != b_min else np.zeros_like(bias)

    quantized_weights = quantize_model(weights_norm,8)
    quantized_bias = quantize_model(bias_norm,8)
   
    weights_rec = dequantize_model(quantized_weights, 8) * (w_max - w_min) + w_min
    bias_rec = dequantize_model(quantized_bias, 8) * (b_max - b_min) + b_min
    bias_rec = bias_rec.item()

    input_dim = weights.shape[0]
    model_torch = SingleLayerPyTorch(input_dim)
    with torch.no_grad():
        model_torch.linear.weight.data = torch.from_numpy(weights_rec.reshape(1, -1).astype(np.float32))
        model_torch.linear.bias.data = torch.tensor([bias_rec], dtype=torch.float32)


    quant_params = {
        'quant_weights': quantized_weights,
        'quant_bias': quantized_bias,
        'w_min': w_min,
        'w_max': w_max,
        'b_min': b_min,
        'b_max': b_max,
        "state_dict": model_torch.state_dict()
    }
    joblib.dump(quant_params, 'models/quant_params.joblib')
   
    print("Original weights shape:", weights.shape)
    print("Quantization error (weights):", np.mean(np.abs(weights - weights_rec)))
    print("Original bias shape:", bias.shape)
    print("Quantization error (bias):", np.mean(np.abs(bias - bias_rec)))
    return model_torch

def final_metrics(sk_model, torch_model):
    data = fetch_california_housing()
    X, y = data.data, data.target

    y_pred_sklearn = sk_model.predict(X)
    r2_sklearn = r2_score(y, y_pred_sklearn)

    torch_model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_pred_torch = torch_model(X_tensor).squeeze().numpy()
    r2_quant = r2_score(y, y_pred_torch)

    unquant_size = os.path.getsize("models/unquant_params.joblib") / 1024
    quant_size = os.path.getsize("models/quant_params.joblib") / 1024

    print(f"R² Score (Original Sklearn):     {r2_sklearn:.4f}")
    print(f"R² Score (Quantized PyTorch):    {r2_quant:.4f}")
    print(f"Model Size (unquant_params.joblib): {unquant_size:.2f} KB")
    print(f"Model Size (quant_params.joblib):   {quant_size:.2f} KB")

if __name__ == "__main__":
    model_path = Path("models/linear_regression.joblib")
    sk_model = load_model(model_path)
    torch_model = run(sk_model)
    final_metrics(sk_model, torch_model)
    torch_model = fix_run(sk_model)
    final_metrics(sk_model, torch_model)
