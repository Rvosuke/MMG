from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# 定义参数搜索空间
param_grid = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'degree': Integer(1, 8),
    'coef0': Real(0, 10)
}

# 使用 Bayesian Optimization 进行超参数搜索
bayes_search = BayesSearchCV(estimator=SVC(), search_spaces=param_grid, n_iter=32, cv=cv, random_state=42, n_jobs=-1)
bayes_search.fit(X_scaled, y)

print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best CV score: {bayes_search.best_score_}")
