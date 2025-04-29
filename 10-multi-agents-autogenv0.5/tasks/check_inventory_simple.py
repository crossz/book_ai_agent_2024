# filename: check_inventory_simple.py

import csv

# 定义库存不足的阈值
THRESHOLD = 10

try:
    # 读取库存文件
    with open('inventory.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        inventory = list(reader)

    # 检查库存并报告不足的鲜花
    low_stock = []
    for item in inventory:
        flower = item['Flower']
        quantity = int(item['Quantity'])
        if quantity < THRESHOLD:
            low_stock.append((flower, quantity))

    # 输出结果
    if low_stock:
        for flower, quantity in low_stock:
            print(f"{flower}: {quantity} 件")
    else:
        print("所有鲜花库存充足。")

except FileNotFoundError:
    print("错误：未找到 'inventory.csv' 文件，请确保文件存在且路径正确。")
except Exception as e:
    print(f"发生错误：{e}")