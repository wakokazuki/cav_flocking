import numpy as np
import datetime
import copy
import sys
import os

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params
from settings import set, show, calc
from agent import agent
from environment import road


def main(parameter: list, max_vel: float, agent_num: int, show_flag=False, return_flag=False):
    """車両追従シミュレーションを実行
    Args:
        max_vel (float): 
        show_flag (bool, optional): _description_. Defaults to False.
        return_flag (bool, optional): _description_. Defaults to False.
    Returns:
        _type_: _description_
    """
    des_dis = calc.get_braking_distance(velocity=max_vel, veh_dis=params.VEH_DIS)
    state_list = agent.StateList(agent_num=agent_num)

    # 初期位置のリストを作成
    init_pos_list = []
    for fol in range(agent_num):
        if params.IS_INTER: # 交差点の場合
            init_pos_list.append(set.get_init_pos_inter(order=fol, des_dis=des_dis))
        else: # 直線の場合
            init_pos_list.append(set.get_init_pos_straight(order=fol, des_dis=des_dis))

    # エージェントのリストの作成
    agents = [agent.Agent(state="C", set_params=None, max_vel=max_vel, init_pos=init_pos_list[i]) for i in range(agent_num)] # 追従車両
        
    # # 追従車両の初期位置
    # for fol in range(agent_num):
    #     if params.IS_INTER: # 交差点の場合
    #         agents[fol].pos[:] = set.get_init_pos_inter(order=fol, des_dis=des_dis) 
    #     else: # 直線の場合
    #         agents[fol].pos[:] = set.get_init_pos_straight(order=fol, des_dis=des_dis)

    # # 位置をコピー
    # for i in range(agent_num):
    #     agents[i].pre_pos = copy.deepcopy(agents[i].pos[:])

    # 目的関数の初期値
    [corr_value, coll_value, obs_value, dis_value, vel_value] = [0.0 for _ in range(len(params.OBJ_LIST))]
    
    # リーダーエージェントの設定
    if params.IS_INTER:
        driving_mode = "inter"
    else:
        driving_mode = "straight"
    leader_path_path = "../out/leader/csv"
    leader_pos = np.loadtxt(f"{leader_path_path}/{driving_mode}_speed{int(max_vel):02}_path_position.csv", delimiter=",", dtype="float64") 
    leader_vel = np.loadtxt(f"{leader_path_path}/{driving_mode}_speed{int(max_vel):02}_path_velocity.csv", delimiter=",", dtype="float64") 
    env = road.Env(driving_mode=driving_mode, n_follower=agent_num, leader_pos=leader_pos.T, leader_vel=leader_vel.T) 


    # 現在時刻 (ファイル名)
    now = datetime.datetime.now()
    date_time = now.strftime("%m%d-%H%M")

    # エージェントシミュレーション
    for t in range(params.NT):
        set.calc_flock(agents=agents, env=env, t=t)
        for num in range(agent_num):
            state_list._append(num=num, vehicle=agents[num]) # 現在の状態をリストに追加
            state_list._check_pos(num=num, vehicle=agents[num], t=t) # 現在の位置を記録
    
    if show_flag:
        timestep = np.arange(params.NT)
        elapsed_time = timestep*params.DT
        # アニメーション
        if params.IS_INTER: # 交差点の場合
            print("Error. intersection animation.")
        else: # 直線の場合
            show.animation_straight(file_name="car_following", date_time=date_time, agents=agents, pos_list=state_list.pos, leader_pos=None, leader_flag=False)
        # 速度
        show.velocity(file_name=date_time, x_list=elapsed_time, y_list=state_list.velocity)
        # 加速度
        show.acceleration(file_name=date_time, x_list=elapsed_time, y_list=state_list.accel)
        # ヨー角
        show.heading_angle(file_name=date_time, x_list=elapsed_time, y_list=state_list.yaw)
        # ステアリング角
        show.steering_angle(file_name=date_time, x_list=elapsed_time, y_list=state_list.delta)
        # 経路
        if params.IS_INTER: # 交差点
            show.inter_path_trajectory(file_name=date_time, pos_list=state_list.pos)
        else: # 直線
            show.straight_path_trajectory(file_name=date_time, pos_list=state_list.pos)

    # 目的関数値を計算
    vel_value = max_vel
    if state_list._judge_pos():
        space_mean_speed = state_list._calc_space_mean_speed(agent_num=agent_num) # [m/s] 空間平均速度
        vel_value = round(max_vel - space_mean_speed, 3) # [m/s] 最大速度と空間平均速度の差分

    # 目的値を出力
    if return_flag: 
        objective_dict = {"CORR":corr_value, "COLL":coll_value, "OBS":obs_value, "DIS":dis_value, "VEL":vel_value}
        return objective_dict
