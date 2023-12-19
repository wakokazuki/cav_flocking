import numpy as np
import datetime

from agent import agent
from settings import set, eval, show, calc
from parameters import params
from environment import road

def main_program(parameter: list, max_vel: float, agent_num: int, show_flag=False, return_flag=False):
    """マルチエージェントシミュレーションを実行
    Args:
        parameter (list): flockingパラメータのリスト
        max_vel (float): エージェントの最大速度
        agent_num (int): エージェントの数
        show_flag (bool, optional): 結果を描画するかどうか
        return_flag (bool, optional): 結果のデータを出力するかどうか
    Returns:
        dict: 目的関数値の辞書
    """
    n_vehicle = params.N_LEADER + agent_num
    des_dis = calc.get_braking_distance(velocity=max_vel, veh_dis=params.VEH_DIS)
    state_list = agent.StateList(agent_num=agent_num)

    # 初期位置のリストを作成
    seed = 25 # 初期位置生成のシード値 
    np.random.seed(seed=seed) 
    init_pos_list = []
    # リーダー
    if params.IS_INTER:
        init_pos_list.append(params.I_POS)
    else:
        init_pos_list.append(params.L_POS)
    # Flocking車両
    for fol in range(agent_num): 
        if params.IS_INTER: # 交差点の場合
            init_pos_list.append(set.get_init_pos_inter(order=fol, des_dis=des_dis))
        else: # 直線の場合
            init_pos_list.append(set.get_init_pos_straight(order=fol, des_dis=des_dis))
            # init_pos_list.append(set.get_init_pos_straight_random(des_dis=des_dis))

    # init_pos_list.extend(params.POS_LIST)

    # # エージェントのリストの作成
    agents = [agent.Agent(state="F", set_params=parameter, max_vel=max_vel, init_pos=init_pos_list[i]) for i in range(n_vehicle)] # Flocking車両
    # リーダーの設定
    if params.VIRTUAL_LEADER:
        agents[params.ORD_LEADER]= agent.Agent(state="V", set_params=None, max_vel=max_vel, init_pos=init_pos_list[params.ORD_LEADER]) # 仮想リーダー
    else:
        agents[params.ORD_LEADER]= agent.Agent(state="L", set_params=None, max_vel=max_vel, init_pos=init_pos_list[params.ORD_LEADER]) # リーダー

    # 目的関数の初期値
    [corr_value, coll_value, obs_value, dis_value, vel_value] = [0.0 for _ in range(len(params.OBJ_LIST))]

    # リーダーエージェントの設定
    if params.IS_INTER:
        driving_mode = "inter"
    else:
        driving_mode = "straight"
    leader_path_path = "../out/leader/csv" #これだと本当に正しいcsvファイルをとってくるかわからないのでは？
    leader_pos = np.loadtxt(f"{leader_path_path}/{driving_mode}_speed{int(max_vel):02}_path_position.csv", delimiter=",", dtype="float64") 
    leader_vel = np.loadtxt(f"{leader_path_path}/{driving_mode}_speed{int(max_vel):02}_path_velocity.csv", delimiter=",", dtype="float64") 
    env = road.Env(driving_mode=driving_mode, n_follower=agent_num, leader_pos=leader_pos.T, leader_vel=leader_vel.T) 

    for veh in range(agent_num):
        if agents[veh].state == "L" or agents[veh].state == "V":
            agents[veh].leader_pos = leader_pos.T
            agents[veh].leader_vel = leader_vel.T

    # 現在時刻 (ファイル名)
    now = datetime.datetime.now()
    date_time = now.strftime("%m%d-%H%M")

    # エージェントシミュレーション
    for t in range(params.NT):
        set.calc_flock(agents, env, t) # 次の状態を計算
        for veh in range(agent_num):
            num = params.N_LEADER + veh # Flocking車両の番号
            state_list._append(num=veh, vehicle=agents[num]) # 現在の状態をリストに追加
            state_list._check_pos(num=veh, vehicle=agents[num], t=t) # 現在の位置を記録
        corr_value += eval.count_corr_value(agents, env)
        coll_value += eval.count_coll_value(agents, env)
        obs_value += eval.count_obs_value(agents, env)
        dis_value += eval.count_dis_value(agents, env)

    if show_flag:
        timestep = np.arange(params.NT)
        elapsed_time = timestep*params.DT
        # アニメーション
        if params.IS_INTER: # 交差点の場合
            #print("Error. intersection animation.")なんでこれがあるの？
            show.animation_inter(file_name="flock", date_time=date_time, pos_list=state_list.pos, leader_pos=leader_pos, leader_flag=True)
        else: # 直線の場合
            show.animation_straight(file_name="flock", date_time=date_time, agents=agents, pos_list=state_list.pos, leader_pos=leader_pos, leader_flag=True)
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
    corr_value = eval.eval_corr_value(corr=corr_value, env=env) 
    dis_value = eval.eval_dis_value(dis=dis_value, env=env)
    vel_value = max_vel
    if state_list._judge_pos():
        space_mean_speed = state_list._calc_space_mean_speed(agent_num=agent_num) # [m/s] 空間平均速度
        vel_value = max_vel - space_mean_speed # [m/s] 最大速度と空間平均速度の差分

    # 目的値を出力
    if return_flag:
        objective_dict = {"CORR":corr_value, "COLL":coll_value, "OBS":obs_value, "DIS":dis_value, "VEL":vel_value}
        return objective_dict
