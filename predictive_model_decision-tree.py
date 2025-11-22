# 定义预测函数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

print("\nClassification Report:")
# 检查实际存在的类别
unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
if len(unique_classes) == 2:
    target_names = ['No Match', 'Match']
else:
    # 如果只有一个类别，根据实际类别设置名称
    target_names = ['No Match' if 0 in unique_classes else 'Match']
classification_report_result = classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes, zero_division=0)
print(classification_report_result)

# 绘制混淆矩阵
print("\n绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
print(f"\n混淆矩阵:\n{cm}")

# 使用 ConfusionMatrixDisplay 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Decision Tree Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 混淆矩阵已保存为 'decision_tree_confusion_matrix.png'")
plt.show()
