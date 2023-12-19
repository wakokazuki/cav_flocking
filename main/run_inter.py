import numpy as np
import datetime
import sys
import os

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent import agent
from settings import set, eval, show, calc
from parameters import params
from environment import road
from path import car_agent

def get_control_input() -> tuple[float, float]:
    pass


def main_program(parameter: list, max_vel: float, agent_num: int, show_flag=False, return_flag=False):
    """マルチエージェントシミュレーションを実行
    Args:
        parameter (list): _description_
        max_vel (float): _description_
        agent_num (int): _description_
        show_flag (bool, optional): _description_. Defaults to False.
        return_flag (bool, optional): _description_. Defaults to False.
    Returns:
        _type_: _description_
    """
    des_dis = calc.get_braking_distance(velocity=max_vel, veh_dis=params.VEH_DIS)
    state_list = agent.StateList(agent_num=agent_num)

    seed = 150 # 初期位置生成のシード値

    # 初期位置のリストを作成
    # init_pos_list = set.get_init_pos_inter_random(seed=seed, agent_num=agent_num)
    init_pos_list = []
    for i in range(params.N_VEHICLE):
        init_pos_list.append(set.get_init_pos_inter(order=i, des_dis=des_dis))
    print(init_pos_list)

    # エージェントのリストの作成
    agents = [agent.Agent(state="F", set_params=parameter, max_vel=max_vel, init_pos=init_pos_list[i]) for i in range(params.N_VEHICLE)] # 追従車両

    # ヨー角・車体の後方位置の更新

    # 経路の生成
    for veh in range(agent_num):
        vehicle = agents[veh]
        vehicle.cx, vehicle.cy = car_agent.get_inter_path(max_vel=max_vel, init_pos=vehicle.pos, final_pos=params.F_POS)
        vehicle.target_course = car_agent.TargetCourse(vehicle.cx, vehicle.cy)

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

    for veh in range(agent_num):
        if agents[veh].state == "L" or agents[veh].state == "V":
            agents[veh].leader_pos = leader_pos.T
            agents[veh].leader_vel = leader_vel.T
            

    # 現在時刻 (ファイル名)
    now = datetime.datetime.now()
    date_time = now.strftime("%m%d-%H%M")

    """
    # エージェントシミュレーション
    for t in range(params.NT):
        # set.calc_flock(agents, env, t) # 次の状態を計算
        for veh in range(agent_num):
            vehicle = agents[veh]
            target_ind, _ = vehicle.target_course.search_target_index(vehicle)
            delta, target_ind = car_agent.pure_pursuit_steer_control(vehicle, vehicle.target_course, target_ind)
            accel = 0.0
            vehicle._update_kbm_rear(accel=accel, delta=delta)
            state_list._append(num=veh, vehicle=agents[veh]) # 現在の状態をリストに追加
            # state_list._check_pos(num=veh, vehicle=agents[num], t=t) # 現在の位置を記録
        # corr_value += eval.count_corr_value(agents, env)
        # coll_value += eval.count_coll_value(agents, env)
        # obs_value += eval.count_obs_value(agents, env)
        # dis_value += eval.count_dis_value(agents, env)"
    """

    # エージェントシミュレーション
    for t in range(params.NT):
        set.calc_flock(agents, env, t) # 次の状態を計算
        for veh in range(agent_num):
            num = params.N_LEADER + veh # Flocking車両の番号
            num = veh
            state_list._append(num=veh, vehicle=agents[num]) # 現在の状態をリストに追加
            state_list._check_pos(num=veh, vehicle=agents[num], t=t) # 現在の位置を記録
        corr_value += eval.count_corr_value(agents, env)
        coll_value += eval.count_coll_value(agents, env)
        obs_value += eval.count_obs_value(agents, env)
        dis_value += eval.count_dis_value(agents, env)


    if show_flag:
        timestep = np.arange(params.NT)
        # アニメーション
        show.animation_inter(file_name="flock", date_time=date_time, agents=agents, pos_list=state_list.pos, leader_pos=None, leader_flag=False)
        # 速度
        # show.velocity(file_name=date_time, x_list=timestep, y_list=state_list.velocity)
        # # 加速度
        # show.acceleration(file_name=date_time, x_list=timestep, y_list=state_list.accel)
        # # ヨー角
        # show.heading_angle(file_name=date_time, x_list=timestep, y_list=state_list.yaw)
        # # ステアリング角
        # show.steering_angle(file_name=date_time, x_list=timestep, y_list=state_list.delta)
        # # 経路
        # show.inter_path_trajectory(file_name=date_time, pos_list=state_list.pos)

    # # 目的関数値を計算
    # corr_value = eval.eval_corr_value(corr=corr_value, env=env) 
    # dis_value = eval.eval_dis_value(dis=dis_value, env=env)

    # 目的値を出力
    if return_flag:
        objective_dict = {"CORR":corr_value, "COLL":coll_value, "OBS":obs_value, "DIS":dis_value, "VEL":vel_value}
        return objective_dict

if __name__ == "__main__":
    main_program(parameter=None, max_vel=params.V_MAX, agent_num=params.N_INTER, show_flag=True, return_flag=True)


