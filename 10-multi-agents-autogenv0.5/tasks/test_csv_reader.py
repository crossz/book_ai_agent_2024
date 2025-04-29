# filename: test_csv_reader.py

import csv

try:
    # 读取库存文件
    with open('inventory.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(row)  # 打印每一行解析后的字典
except FileNotFoundError:
    print("错误：未找到 'inventory.csv' 文件，请确保文件存在且路径正确。")
except Exception as e:
    print(f"发生错误：{e}")