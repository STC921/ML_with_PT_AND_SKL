import csv
import re

# 定义输入数据文件和输出CSV文件的名称
input_file = "output.txt"
output_csv = "results.csv"

# 打开输入数据文件
with open(input_file, 'r') as f:
    data = f.read()

# 确认读取的数据内容
print("Data read from file:")
print(data[:1000])  # 打印前1000个字符以检查内容

# 定义正则表达式模式
pattern = re.compile(r"Running with (\d+) threads.*?Mat-Mat time  = (\d+\.\d+) \[sec.\].*? (\d+\.\d+) \[MFLOPS\]", re.DOTALL)

# 搜索匹配项
matches = pattern.findall(data)

# 确认匹配结果
print("Matches found:")
print(matches)

# 将数据写入CSV文件
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Threads', 'Mat-Mat Time (sec)', 'MFLOPS'])  # 写入表头
    for match in matches:
        csv_writer.writerow(match)

print(f"Data has been written to {output_csv}")
