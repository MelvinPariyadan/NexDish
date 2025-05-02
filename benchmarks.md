 NexDish Model Evaluation Benchmarks

This file outlines the performance benchmarks and evaluation criteria used to guide the development and testing of the NexDish smart meal recognition model.


Primary Goals

| Metric                     | Target Value                  | Why It Matters                                           |
|----------------------------|-------------------------------|----------------------------------------------------------|
| **Top-1 Accuracy**         | ≥ 85%                         | Ensures high likelihood that the first prediction is correct |
| **Inference Time**         | ≤ 0.5 seconds per image       | Allows real-time feedback in mobile/web environments     |
| **Train vs Validation Gap**| ≤ 5%                          | Ensures generalization and avoids overfitting            |
| **Cultural Diversity**     | Broad coverage of cuisines    | Must perform well across global food categories          |
| **Robustness**             | Handles blurry or mixed images| Improves real-world usability and reduces errors         |



Inference Time Test Code Snippet

```python
import time

model.eval()
sample = next(iter(val_loader))
img = sample['pixel_values'][0].unsqueeze(0).to(device)

start = time.time()
_ = model(img)
end = time.time()

print(f"Inference time: {end - start:.4f} seconds")
