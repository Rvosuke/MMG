# 👀 Initial-Glau Screening Model

This directory provides the Python code for implementing the initial screening model. The model utilizes traditional machine learning algorithms (SVM, XGBoost, LR, KNN) to perform glaucoma screening based on preliminary medical data.

## 🚀 Features

- 🔍 Glaucoma screening using basic medical data (age, gender, BCVA, IOP, CDR)
- 🛠️ Implementation of SVM, XGBoost, LR, and KNN models
- ⚙️ Hyperparameter optimization with Bayesian Optimization (BO)
- 🔬 Model interpretability analysis using SHAP [(SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/index.html#)

## :book:Performance

| Model   | Accuracy   | Sensitivity | Specificity | AUC        |
| ------- | ---------- | ----------- | ----------- | ---------- |
| KNN     | 0.6453     | 0.6303      | 0.6682      | 0.7155     |
| SVM     | 0.7697     | 0.8273      | 0.6820      | 0.8486     |
| LR      | 0.7697     | 0.8364      | 0.6682      | 0.8488     |
| XGBoost | **0.8208** | **0.8394**  | **0.7926**  | **0.8929** |