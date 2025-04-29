# filename: check_file.py

import os

# 检查文件是否存在
if os.path.exists('inventory.csv'):
    print("文件存在，内容如下：")
    with open('inventory.csv', mode='r') as file:
        print(file.read())
else:
    print("文件 'inventory.csv' 不存在。请确保文件与脚本在同一目录下。")