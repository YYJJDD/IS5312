# 定义预测函数
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from read_csv_data import df_clean

X = df_clean.drop('match', axis=1)
# 目标变量
y = df_clean['match']

# 8:2分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集大小：{len(X_train)}, 测试集大小：{len(X_test)}")

def train_predict_model(X_train, y_train, X_test, model=DecisionTreeClassifier(random_state=42)):
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    return model, y_pred

dt_model, y_pred = train_predict_model(X_train, y_train, X_test)

print("dt_model:", dt_model)
print("y_pred:", y_pred)
