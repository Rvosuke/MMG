# 👀 Initial-Glau Screening Model

[![stars](https://img.shields.io/github/stars/Rvosuke/MMG?style=social)](https://github.com/yourusername/repo)
[![forks](https://img.shields.io/github/forks/Rvosuke/MMG?style=social)](https://github.com/yourusername/repo)
[![issues](https://img.shields.io/github/issues/Rvosuke/MMG)](https://github.com/yourusername/repo/issues)
[![license](https://img.shields.io/github/license/Rvosuke/MMG)](https://github.com/yourusername/repo/blob/main/LICENSE)

This repository provides the Python code for implementing the initial screening model proposed in the paper "A Machine Learning Framework for Grading Glaucoma Severity for the Tiered Healthcare System in China". The model utilizes traditional machine learning algorithms (SVM, XGBoost, LR, KNN) to perform glaucoma screening based on preliminary medical data.

## 🚀 Features

- 🔍 Glaucoma screening using basic medical data (age, gender, BCVA, IOP, CDR)
- 🛠️ Implementation of SVM, XGBoost, LR, and KNN models
- ⚙️ Hyperparameter optimization with Bayesian Optimization (BO)
- 🔬 Model interpretability analysis using SHAP [(SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/index.html#)
- :hugs:  All models can be test on [huggingface](https://huggingface.co/spaces/Aohanah/Window)

| Model   | Accuracy   | Sensitivity | Specificity | AUC        |
| ------- | ---------- | ----------- | ----------- | ---------- |
| KNN     | 0.6453     | 0.6303      | 0.6682      | 0.7155     |
| SVM     | 0.7697     | 0.8273      | 0.6820      | 0.8486     |
| LR      | 0.7697     | 0.8364      | 0.6682      | 0.8488     |
| XGBoost | **0.8208** | **0.8394**  | **0.7926**  | **0.8929** |

## 📂 Repository Structure

```
├── data/
│   └── ... (dataset files not public)
├── models/
│   ├── svm.py
│   ├── xgboost.py
│   ├── lr.py
│   └── knn.py
├── src/
│   ├── preprocessing.py
│   ├── optimization.py
│   └── visualization.py
├── main.py
└── README.md
```

## 🏃‍♂️ Getting Started

1. Clone the repository:
```
git clone https://github.com/Rvosuke/MMG.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the main script:
```
python main.py
```

## 🤝 Contributing

Contributions are always welcome! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the [MIT License](https://github.com/yourusername/repo/blob/main/LICENSE).

Due to healthcare data management policies, the data used in this study cannot be made publicly accessible. However, the Multi-Glau discussed in this paper are generic and can be used as long as the input consists of medical images and structured numerical data.

## 🙏 Acknowledgments

This project is based on [Multi-Glau](https://github.com/CowboyH/Multi-Glau). Very thanks to the authors of the paper "A Machine Learning Framework for Grading Glaucoma Severity for the Tiered Healthcare System in China" for their valuable research contribution.
