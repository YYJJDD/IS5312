# 品质重要性相关列
from matplotlib import pyplot as plt

from read_csv_data import df_clean

important_cols = ['attractive_important', 'sincere_important', 'intelligence_important',
                 'funny_important', 'ambition_important', 'shared_interests_important']
partner_cols = ['attractive_partner', 'sincere_partner', 'intelligence_partner',
               'funny_partner', 'ambition_partner', 'shared_interests_partner']

# 第一个参与者的品质重要性饼图
first_participant = df_clean.iloc[0]
important_values = first_participant[important_cols].values

plt.figure(figsize=(8, 8))
plt.pie(important_values, labels=important_cols, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
plt.title('six qualities importance of the first participant', fontsize=16)
plt.axis('equal')
plt.show()
plt.savefig('six_qualities_importance_of_the_first_participant_pie_chart.png', bbox_inches='tight')

# 计算相对分数并添加到数据框
for imp_col, par_col in zip(important_cols, partner_cols):
    rel_col = f'relative_{imp_col.split("_")[0]}'
    df_clean[rel_col] = df_clean[par_col] * (df_clean[imp_col] / 100)