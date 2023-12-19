import time
import concurrent.futures
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import copy
import glob

# root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from parameters import params
from agent import agent
from settings import set, calc
from environment import road
from main import run

def get_heatmap_csv(two_parameters):
    # 最新のCSVファイルを選択
    list_of_files = glob.glob("../out/csv/*.csv")
    latest_file = max(list_of_files, key=os.path.getctime)

    row_num = params.N_ROW # パラメータのセットの行番号 
    df = pd.read_csv(latest_file, header=0, index_col=0)
    [parameter] = df[row_num:row_num+1].values.tolist()
    heatmap_vel = two_parameters[0]
    heatmap_agent_num = two_parameters[1]
    # # Flockingパラメータの調整
    optimized_vel = params.V_MAX
    # r0_rep[m]
    order = 0
    heatmap_des_dis = calc.get_braking_distance(velocity=heatmap_vel, veh_dis=params.VEH_DIS)
    heatmap_cls_dis = np.sqrt(2)*heatmap_des_dis
    ratio = (parameter[order] - params.DES_DIS) / (params.CLS_DIS - params.DES_DIS)
    parameter[order] = ratio * (heatmap_cls_dis - heatmap_des_dis) + heatmap_des_dis
    # r0_frict[m]
    order = 2
    ratio = ((parameter[order] - params.DES_DIS/2) / (params.DES_DIS/2))
    parameter[order] = ratio * (heatmap_des_dis/2) + heatmap_des_dis/2
    # v_frict[m]
    order = 4
    parameter[order] *= heatmap_vel / optimized_vel
    # 目的関数値を取得
    objective_dict = run.run_main_program(parameter=parameter, max_vel=heatmap_vel, agent_num=heatmap_agent_num, return_flag=True)
    vel_value = objective_dict["VEL"] # [m/s] 最大速度と空間平均速度の差分
    return vel_value

if __name__ == "__main__":
    # 開始時刻の記録
    start = time.perf_counter()
    # ヒートマップのパラメータ
    x_num = params.HEAT_X_NUM
    y_num = params.HEAT_Y_NUM
    x_limit = [params.HEAT_V_MIN, params.HEAT_V_MAX]
    y_limit = [params.HEAT_AGENT_MIN, params.HEAT_AGENT_MAX]
    # ループ処理
    for i in range(10):
        n_row = i
        heat_matrix = set.get_heatmap_matrix_loop(x_num=x_num, y_num=y_num, x_limit=x_limit, y_limit=y_limit, n_row=n_row)
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
        now = datetime.now()
        video_name = now.strftime("%m%d-%H%M") 
        df.to_csv(f"../out/heatmap/csv/heatmap_{video_name}_loop{n_row:02}_row{y_num}_column{x_num}_space_mean_speed.csv")
    # 計算時間を出力
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")
