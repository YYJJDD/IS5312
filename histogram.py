# 筛选数值变量（排除非数值类型列）
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from extract_variables_column import var_names
from read_csv_data import df_clean

df = pd.read_csv('SpeedDating.csv', header=None)
# 添加列名
df.columns = var_names

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 描述性统计
desc_stats = df_clean[numeric_cols].describe()
print("数值变量描述性统计：")
print(desc_stats)

# 绘制直方图（每页显示6个图）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
fig, axes = plt.subplots(nrows=(len(numeric_cols)//6)+1, ncols=6, figsize=(20, 3*(len(numeric_cols)//6)+3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        df_clean[col].hist(ax=axes[i], bins=20, edgecolor='black')
        axes[i].set_title(f'{col} distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('frequency')

# 隐藏多余子图
for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

plt.savefig('histogram.png')
