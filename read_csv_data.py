# 读取数据，无表头
import pandas as pd

from extract_variables_column import var_names

df = pd.read_csv('SpeedDating.csv', header=None)
# 添加列名
df.columns = var_names[:len(df.columns)]  # 确保列名数量匹配

# 删除缺失值
df_clean = df.dropna()
print(f"删除缺失值后剩余观测数：{len(df_clean)}")

# 转换gender为二进制（1=male，0=female）
df_clean['gender'] = df_clean['gender'].apply(lambda x: 1 if x == b'male' else 0)

# 转换samerace和match为二进制
def convert_binary(x):
    return 1 if x == b'1' else 0

df_clean['samerace'] = df_clean['samerace'].apply(convert_binary)
df_clean['match'] = df_clean['match'].apply(convert_binary)

# 保存为新CSV
df_clean.to_csv('dataforanalysis.csv', index=False)