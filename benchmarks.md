 NexDish Model Evaluation Benchmarks

This file outlines the performance benchmarks and evaluation criteria used to guide the development and testing of the NexDish smart meal recognition model.


Primary Goals

| Metric               | Target Value            | Why It Matters                                                                                         |
|----------------------|------------------------|--------------------------------------------------------------------------------------------------------|
| **Top-1 Accuracy**   | ≥ 85%                  | Ensures high likelihood that the first prediction is correct                                           |
| **Inference Time**   | ≤ 0.2 seconds/image    | Allows real-time feedback in mobile/web environments                                                   |
| **End-to-End Testing** | Input on frontend returns correct final output after all required backend services are called | Validates that the full user flow works as expected, including integration between frontend and all dependent backend/model services; ensures reliable user experience and early detection of integration issues |


