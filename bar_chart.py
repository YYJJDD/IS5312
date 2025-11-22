# 计算年龄差（绝对值）
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from read_csv_data import df_clean

df_clean['age_d'] = abs(df_clean['age'] - df_clean['age_o'])

# 年龄差分组（0-1, 2-3, ..., 按实际范围调整）
age_d_bins = np.arange(0, df_clean['age_d'].max()+2, 2)
age_d_groups = pd.cut(df_clean['age_d'], bins=age_d_bins, right=False)
age_d_count = age_d_groups.value_counts().sort_index()

# 绘制条形图
plt.figure(figsize=(12, 6))
age_d_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('年龄差分布')
plt.xlabel('年龄差分组')
plt.ylabel('观测数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()