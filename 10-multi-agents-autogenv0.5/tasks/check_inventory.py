# filename: check_inventory.py

import csv

# 定义库存不足的阈值
THRESHOLD = 10

# 读取库存数据
try:
    with open('inventory.csv', mode='r') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ['Flower', 'Quantity']:
            print("错误：'inventory.csv' 文件格式不正确。请确保文件包含 'Flower' 和 'Quantity' 列。")
        else:
            inventory = list(reader)

            # 检查库存并报告不足的鲜花
            low_stock = []
            for item in inventory:
                flower = item['Flower']
                quantity = int(item['Quantity'])
                if quantity < THRESHOLD:
                    low_stock.append((flower, quantity))

            # 输出库存不足的鲜花
            if low_stock:
                print("以下鲜花库存不足：")
                for flower, quantity in low_stock:
                    print(f"{flower}: {quantity} 件")
            else:
                print("所有鲜花库存充足。")
except FileNotFoundError:
    print("错误：未找到 'inventory.csv' 文件。请确保文件存在。")