import os 
import time
import sys

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params
import run_car_following

if __name__ == "__main__":
    # 開始時刻を記録
    start = time.perf_counter()
    # 目的関数値を計算 
    objective_dict = run_car_following.main(parameter=None, max_vel=params.V_MAX, agent_num=params.N_FOLLOWER, show_flag=True, return_flag=True)
    print(objective_dict)
    # 経過時間を出力
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")
