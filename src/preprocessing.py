import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


# 加载数据集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 检查缺失值
print(df.isnull().sum())

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[data.feature_names])
y = df['target']
