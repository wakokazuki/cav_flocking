import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import csv
import pathlib
import operator 
from operator import itemgetter

#csvファイルの選択基準
method_label = "ctaea"
pop_num = 120

#csvファイルの選択
list_of_files = glob.glob(f'../out/csvii/*/*/*')
for file in list_of_files:
    print(file)

#csvファイルの並び替えの関数
def sort_csv(csv_file: pathlib, sort_row1: int):
    # 今回作成したuser_list_csvを開く
    csv_data = csv.reader(open(csv_file), delimiter=',')
    # ヘッダー情報を取得
    header = next(csv_data)
    # ヘッダー以外の列を並び替える
    sort_result = sorted(csv_data,reverse=True ,key=itemgetter(sort_row1))
    #print(sort_result)
    # 新規ファイルとしてuser_list_csvを開く
    with open(csv_file, "w") as f:
        # ヘッダーと並び替え結果をファイルに書き込む
        data = csv.writer(f, delimiter=',')
        data.writerow(header)
        for r in sort_result:
            data.writerow(r)

#csvファイルの並び替えの繰り返し
for file in list_of_files:
    sort_csv(file,13)
for file in list_of_files:
    sort_csv(file,14)