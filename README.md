# mlops-assignment-3



### Model Comparison Metrics

|------------------------------|-----------------------|-------------------------------|
| Metric                       | Sklearn Model         | Quantized PyTorch Model       |
|------------------------------|-----------------------|-------------------------------|
| R² Score                     | 0.6053                | -619043.1550                  |
| Model Size (`.joblib`)       | 0.40 KB               | 1.24 KB                       |
|------------------------------|-----------------------|-------------------------------|


The negative R² score for the quantized model suggests a severe mismatch — likely due to quantization precision loss, possibly from very small weight ranges or inappropriate dequantization.

|------------------------------|-----------------------|-------------------------------|
| Metric                       | Sklearn Model         | Quantized PyTorch Model       |
|------------------------------|-----------------------|-------------------------------|
| R² Score                     | 0.6053                | -0.1542                       |
| Model Size (`.joblib`)       | 0.40 KB               | 1.25 KB                       |
|------------------------------|-----------------------|-------------------------------|
