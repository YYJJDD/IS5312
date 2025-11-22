### 3.2.2 计算年龄差并绘制条形图（不分组，展示每个年龄差）（15%）
# 1. 计算年龄差（绝对值：避免正负抵消）
from matplotlib import pyplot as plt

from read_csv_data import df_clean

df_clean['age_d'] = abs(df_clean['age'] - df_clean['age_o'])
print(f"\n1. 年龄差（age_d）计算完成，范围：{df_clean['age_d'].min():.0f} ~ {df_clean['age_d'].max():.0f} 岁")

# 2. 统计每个具体年龄差的观测数（不分组）
age_d_count = df_clean['age_d'].value_counts().sort_index()  # 按年龄差数值排序
print(f"\n2. 各年龄差观测数统计：")
print(age_d_count)

# 3. 绘制条形图（不分组，每个年龄差单独展示）
plt.figure(figsize=(14, 7))
age_d_count.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.8, width=0.8)
plt.title('Age Difference Distribution', fontsize=16)
plt.xlabel('Age Difference', fontsize=14)
plt.ylabel('Number of Observations', fontsize=14)
plt.xticks(rotation=0)  # 年龄差标签水平显示，不旋转
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('age_difference_bar_ungrouped.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\n3. 不分组的年龄差条形图已保存为：age_difference_bar_ungrouped.png")