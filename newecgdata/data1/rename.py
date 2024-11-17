# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:18:18 2024

@author: Zeff
"""

import os
import pandas as pd

# 读取Diagnostics.xlsx文件并命名为table1
table1 = pd.read_excel('Diagnostics.xlsx')

# 获取当前目录下的所有csv文件的文件名（不包括扩展名）
csv_files = [os.path.splitext(f)[0] for f in os.listdir('.') if f.endswith('.csv')]

# 遍历每个csv文件（仅基本名称）
for csv_file in csv_files:
    # 在table1中查找该csv文件的文件名（不包括扩展名）
    row = table1[table1.iloc[:, 0] == csv_file]
    
    # 如果找到该文件名
    if not row.empty:
        # 获取该行的第三列的值
        third_column_value = row.iloc[0, 2]
        
        # 判断第三列的值
        if third_column_value == 'NONE':
            # 重命名文件，在原文件名前插入N
            new_name = 'NONE' + csv_file + '.csv'
        else:
            # 重命名文件，在原文件名前插入O
            new_name = 'OTHER' + csv_file + '.csv'
        
        # 重命名文件，保留扩展名.csv
        os.rename(csv_file + '.csv', new_name)
        print(f'Renamed {csv_file}.csv to {new_name}')
    else:
        print(f'File {csv_file} not found in Diagnostics.xlsx')

print('Renaming completed!')
