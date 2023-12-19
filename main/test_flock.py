import pandas as pd
import os 
import glob
import time
import sys

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params
import run as run

if __name__ == "__main__":
    # 開始時刻を記録
    start = time.perf_counter()
    # 最新のCSVファイルを選択
    list_of_files = glob.glob("../out/csv/case7.csv")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)

    # select CSV file
    # selected_file = ""

    row_num = params.N_ROW # csvの行番号 
    df = pd.read_csv(latest_file, header=0, index_col=0)

    [parameter] = df[row_num:row_num+1].values.tolist()
    print(parameter)
    objective_dict = run.main_program(parameter=parameter, max_vel=params.V_MAX, agent_num=params.N_FOLLOWER, show_flag=True, return_flag=True)
    print(objective_dict)
    # 経過時間を出力
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")