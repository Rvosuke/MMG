from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

# 定义 SVM 模型
svm_model = SVC()

# 使用 10-fold 交叉验证
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_model, X_scaled, y, cv=cv)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean()}")
