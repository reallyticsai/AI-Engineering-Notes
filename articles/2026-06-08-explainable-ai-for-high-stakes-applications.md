---
tags: explainable-ai, medical-diagnosis, shap, lime, model-interpretability
---

# Unboxing Black Boxes: Using SHAP and LIME to Explain Medical Diagnosis Models
By Reallytics AI

## TL;DR
* Explainable AI (XAI) is crucial for high-stakes applications like medical diagnosis, where model interpretability is as important as accuracy.
* SHAP and LIME are two leading post-hoc explainability techniques that help unbox black-box models, providing insights into their decision-making processes.
* This article dives into the technical details of SHAP and LIME, with code examples and practical lessons learned from production experiences.

## Introduction
The increasing use of AI in medical diagnosis has raised concerns about the interpretability of complex models. While deep neural networks and ensemble methods have shown remarkable accuracy in diagnosing diseases, their lack of transparency can be a significant barrier to adoption in clinical settings. Explainable AI (XAI) has emerged as a critical field, aiming to provide insights into the decision-making processes of these black-box models. In this article, we'll explore two popular XAI techniques: SHAP and LIME, and demonstrate their application in explaining medical diagnosis models.

## Technical Deep Dive
### SHAP: SHapley Additive exPlanations
SHAP is a unified framework for additive feature attribution methods, based on Shapley values from cooperative game theory. It provides both global and local explanations for model predictions. Here's an example of using SHAP to explain an XGBoost model trained on the UCI Breast Cancer Wisconsin (Diagnostic) dataset:
```python
import xgboost as xgb
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot SHAP values for a specific instance
shap.plots.waterfall(shap_values[0], max_display=10)
```
This code generates a waterfall plot showing the contribution of each feature to the predicted output for a specific instance.

### LIME: Local Interpretable Model-agnostic Explanations
LIME is a perturbation-based approach that locally approximates complex models with interpretable surrogate models. Here's an example of using LIME to explain a PyTorch neural network trained on the same dataset:
```python
import torch
import torch.nn as nn
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Define PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train PyTorch model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
    loss.backward()
    optimizer.step()

# Create LIME explainer
explainer = LimeTabularExplainer(X_train, feature_names=data.feature_names, class_names=data.target_names, discretize_continuous=True)

# Explain a specific instance
exp = explainer.explain_instance(X_test[0], lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()[:, 1], num_features=10)
exp.as_pyplot_figure().show()
```
This code generates a bar plot showing the contribution of each feature to the predicted output for a specific instance.

### Architecture Diagram
Our XAI pipeline can be represented as follows:
```
                      +---------------+
                      |  Medical Data  |
                      +---------------+
                             |
                             |
                             v
                      +---------------+
                      |  Model Training  |
                      |  (XGBoost/PyTorch) |
                      +---------------+
                             |
                             |
                             v
                      +---------------+
                      |  SHAP/LIME      |
                      |  Explanation     |
                      +---------------+
                             |
                             |
                             v
                      +---------------+
                      |  Insights and   |
                      |  Interpretability|
                      +---------------+
```
This diagram illustrates the flow of data through our pipeline, from data ingestion to model training and explanation.

## Production Lessons Learned
In our experience deploying XAI in production environments, we've learned the following key lessons:
* **Model complexity is not a substitute for interpretability**: While complex models may offer better accuracy, they can be difficult to interpret, making it challenging to trust their predictions.
* **SHAP and LIME are complementary techniques**: SHAP provides a more comprehensive understanding of feature importance, while LIME offers a local, instance-level explanation.
* **Explainability is not a one-time task**: As models evolve and new data becomes available, explanations need to be recomputed to ensure that insights remain relevant.

## Key Takeaways
* SHAP and LIME are powerful XAI techniques for explaining complex medical diagnosis models.
* By providing insights into model decision-making processes, XAI can increase trust and adoption in clinical settings.
* Production deployments require careful consideration of model complexity, interpretability, and ongoing explanation.

## Further Reading
For more information on SHAP and LIME, we recommend the following resources:
* [SHAP documentation](https://shap.readthedocs.io/en/latest/index.html)
* [LIME GitHub repository](https://github.com/marcotcr/lime)
* [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/index.html)