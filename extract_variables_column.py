

# 读取variables.txt，跳过前2行，提取变量名（假设每行第一列为变量名）
with open('variables.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
# 跳过前2行，分割每行取第一列作为变量名，去除空格和换行符
var_names = [line.strip().split(' ')[0] for line in lines[5:]]  # 假设列分隔符为制表符，若为其他可调整

print(var_names)