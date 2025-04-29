# filename: check_inventory.py

import csv

# 读取库存文件
with open('inventory.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    inventory = list(csv_reader)

# 定义库存不足的标准
threshold = 10

# 检查库存不足的鲜花
low_stock = []
for item in inventory:
    if int(item['Quantity']) < threshold:
        low_stock.append((item['Flower'], item['Quantity']))

# 输出结果
if low_stock:
    print("以下鲜花库存不足：")
    for flower, quantity in low_stock:
        print(f"{flower}: 当前库存 {quantity}")
else:
    print("所有鲜花库存充足。")