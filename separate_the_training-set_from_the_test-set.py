# 特征（排除目标变量match）
from sklearn.model_selection import train_test_split

from read_csv_data import df_clean

X = df_clean.drop('match', axis=1)
# 目标变量
y = df_clean['match']

# 8:2分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集大小：{len(X_train)}, 测试集大小：{len(X_test)}")