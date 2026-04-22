import re

# 输入文件路径
input_file = 'data2.txt'
# 输出文件路径
output_file = 'data2_cleaned.txt'

# 要移除的模式
patterns_to_remove = [
    r'【第\d+页】',  # 页码格式，如【第17页】
    r'\(\d+\)',      # 数字括号，如(10), (020)
    r'\(\d+',        # 不完整的数字括号，如(011
    r'\d+',          # 单独的数字，如012
]

# 合并所有模式为一个正则表达式
combined_pattern = '|'.join(patterns_to_remove)

# 读取文件并清洗
with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# # 筛选行：只保留 568 到 5798 行（注意：Python 索引从 0 开始）
# start_line = 567  # 568 行对应索引 567
# end_line = 5797   # 5798 行对应索引 5797
# filtered_lines = lines[start_line:end_line+1]

# 清洗每一行
cleaned_lines = []
for line in lines:
    # 移除指定模式
    cleaned_line = re.sub(combined_pattern, '', line)
    # 移除多余的空白字符
    cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
    # 只添加非空行
    if cleaned_line:
        cleaned_lines.append(cleaned_line + '\n')

# 回车换行太多了，搞一个只智能换行的函数
def smart_newline(text):
    return re.sub(r'\n+', '', text)

# 应用智能换行函数到cleaned_lines
cleaned_lines = [smart_newline(line) for line in cleaned_lines]

# 写入清洗后的文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"清洗完成！")
print(f"原始行数: {len(lines)}")
print(f"清洗后行数: {len(cleaned_lines)}")
print(f"清洗后的文件已保存至: {output_file}")


