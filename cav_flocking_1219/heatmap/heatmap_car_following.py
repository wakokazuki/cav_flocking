import time
import concurrent.futures
import pandas as pd
import numpy as np
import datetime
import os
import sys

# root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from parameters import params
from settings import set
from car_following_model import run_car_following

def get_heatmap_csv(two_parameters): # 最大速度, 車両台数に応じた指標値を取得
    heatmap_vel = two_parameters[0]
    heatmap_agent_num = two_parameters[1]
    objective_dict = run_car_following.main(parameter=None, max_vel=heatmap_vel, agent_num=heatmap_agent_num, return_flag=True)
    vel_value = objective_dict["VEL"] # [m/s] 最大速度と空間平均速度の差分
    return vel_value


# ヒートマップのラベル
x_param_label = "max speed (m/s)"
y_param_label = "vehicle number"

if __name__ == "__main__":
    """速度低下指標(最大速度-空間平均速度)のヒートマップを作成
    """
    # 開始時刻の記録
    start = time.perf_counter()
    # ヒートマップのパラメータ
    x_num = params.HEAT_X_NUM
    y_num = params.HEAT_Y_NUM
    x_limit = [params.HEAT_V_MIN, params.HEAT_V_MAX]
    y_limit = [params.HEAT_AGENT_MIN, params.HEAT_AGENT_MAX]
    heat_matrix = set.get_heatmap_matrix(x_num=x_num, y_num=y_num, x_limit=x_limit, y_limit=y_limit)
    # DataFrameを作成
    pd_index, pd_columns = [], []
    for row in range(y_num):
        pd_index.append(round(heat_matrix[row][0][1], 1))
    for col in range(x_num):
        pd_columns.append(round(heat_matrix[0][col][0], 1))
    # # マルチプロセシングの入力データの作成
    input_matrix = [x for row in heat_matrix for x in row] # 2次元リストから1次元のリストへ変換
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(get_heatmap_csv, input_matrix)
    results_list = []
    for r in results:
        results_list.append(r)
    df = pd.DataFrame(np.array(results_list).reshape(y_num, x_num), columns=pd_columns, index=pd_index)
    # CSVファイルを保存
    now = datetime.datetime.now()
    video_name = now.strftime("%m%d-%H%M") 
    df.to_csv(f"../out/heatmap/csv/heatmap_{video_name}_row{y_num}_column{x_num}_car_following.csv")
    # 計算時間を出力
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")
