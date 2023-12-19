import os
import sys
import numpy as np

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params
from path import car_agent

def get_lane_index(pos_y: float) -> int:
    """現在の車線の番号を取得
    Args:
        pos_y (float): [m] 車両の位置座標(y軸)
    Returns:
        int: 車線の番号(0,1,2,...)
    """
    for lane in range(params.N_LANE):
        if lane*params.W_LANE <= pos_y <= (lane+1)*params.W_LANE:
            return lane

def get_lead_name(agents: list, vehicle_index: int) -> str:
    """車両の前方オブジェクトの名前を取得
    Args:
        agents (list): 車両エージェントのリスト
        vehicle_index (int): 車両の番号
    Returns:
        str: 前方オブジェクトの名前("obstacle" or "vehicle" or "none")
    """
    vehicle = agents[vehicle_index] # 車両
    lane_index = get_lane_index(pos_y=vehicle.pos[1]) # 車線の番号
    # 前方の車両との距離
    veh_dis = params.CAV_DIS
    for veh in range(len(agents)):
        if veh == vehicle_index:
            continue
        veh_lane_index = get_lane_index(pos_y=agents[veh].pos[1]) # 他の車両の車線番号
        if agents[veh].state != "V" and lane_index == veh_lane_index and vehicle.pos[0] <= agents[veh].pos[0]:
            veh_dis = min(veh_dis, agents[veh].pos[0] - vehicle.pos[0])
    # 前方の障害物との距離
    obs_dis = params.CAV_DIS
    for keys in params.OBS_DICT.keys():
        if params.OBS_DICT[keys]:
            obs_index = get_lane_index(pos_y=params.OBS_POS_DICT[keys][1]) # 障害物の車線番号
            obs_lane_change_dis = params.OBS_POS_DICT[keys][0] - params.LANE_RATE
            if lane_index == obs_index and vehicle.pos[0] <= obs_lane_change_dis:
                obs_dis = min(obs_dis, obs_lane_change_dis - vehicle.pos[0])
    # 前方の種類を取得
    if veh_dis < obs_dis:
        lead_name = "vehicle"
    elif veh_dis > obs_dis:
        lead_name = "obstacle"
    else:
        lead_name = "none"
    return lead_name

def get_lead_gap(agents: list, vehicle_index: int, lane_index: int) -> float:
    """車両の前方ギャップを取得
    Args:
        agents (list): 車両エージェントのリスト
        vehicle_index (int): 車両の番号
        lane_index (int): 車線の番号
    Returns:
        float: [m] 車両の前方距離
    """
    vehicle = agents[vehicle_index] # 車両
    # 前方の車両との距離
    veh_dis = params.CAV_DIS
    for veh in range(len(agents)):
        if veh == vehicle_index:
            continue
        veh_lane_index = get_lane_index(pos_y=agents[veh].pos[1]) # 他の車両の車線番号
        if agents[veh].state != "V" and lane_index == veh_lane_index and vehicle.pos[0] <= agents[veh].pos[0]:
            veh_dis = min(veh_dis, agents[veh].pos[0] - vehicle.pos[0])
    # 前方の障害物との距離
    obs_dis = params.CAV_DIS
    for keys in params.OBS_DICT.keys():
        if params.OBS_DICT[keys]:
            obs_index = get_lane_index(pos_y=params.OBS_POS_DICT[keys][1]) # 障害物の車線番号
            obs_lane_change_dis = params.OBS_POS_DICT[keys][0] - params.LANE_RATE
            if lane_index == obs_index and vehicle.pos[0] <= obs_lane_change_dis:
                obs_dis = min(obs_dis, obs_lane_change_dis - vehicle.pos[0])
    # 前方の距離を取得
    lead_gap = min(veh_dis, obs_dis)
    return lead_gap

def get_lag_gap(agents: list, vehicle_index: int, lane_index: int) -> float:
    """車両の後方ギャップを取得
    Args:
        agents (list): 車両エージェントのリスト
        vehicle_index (int): 車両の番号
        lane_index (int): 車線の番号
    Returns:
        float: [m] 車両の後方距離
    """
    vehicle = agents[vehicle_index] # 車両
    # 後方の車両との距離
    veh_dis = vehicle.des_dis*2.5
    for veh in range(len(agents)):
        if veh == vehicle_index:
            continue
        veh_index = get_lane_index(pos_y=agents[veh].pos[1]) # 他の車両の車線番号
        if lane_index == veh_index and vehicle.pos[0] >= agents[veh].pos[0]:
            veh_dis = min(veh_dis, vehicle.pos[0] - agents[veh].pos[0])
    # 後方の距離を取得
    lag_gap = veh_dis
    return lag_gap


def optimal_velocity_model(vehicle: object, lead_dis: float, pre_velocity: float) -> float:
    """最適速度モデルによる加速度の計算 # 式は次の論文を参照(https://doi.org/10.1103/PhysRevE.58.5429)
    Args:
        lead_dis (float): [m] 車両の前方の距離
        pre_velocity (float): 車両の現在速度
    Returns:
        float: [m/s/s] 加速度
    """
    def _v(dis: float, safe_dis: float, v_max: float) -> float:
        """最適速度関数"""
        velocity = v_max/2*(np.tanh(dis - safe_dis) + np.tanh(safe_dis))
        return velocity
    # パラメータ
    alpha = 1.0
    accel = alpha*(_v(dis=lead_dis, safe_dis=vehicle.des_dis, v_max=vehicle.max_vel) - pre_velocity)
    return accel

def judge_lane_change(vehicle: object, lead_gap: float, lag_gap: float, vel: float=None, read_vel: float=None, lag_vel: float=None) -> bool:
    min_lead_gap = vehicle.des_dis # [m]
    min_lag_gap = vehicle.des_dis*2.5 # [m]
    # 距離間隔判定
    if lead_gap >= min_lead_gap and lag_gap >= min_lag_gap:
        is_lane_change = True
    else:
        is_lane_change = False
    return is_lane_change

def judge_driving_mode(lead_name: str, agents: list, vehicle_index: int) -> str:
    def _decide_lane(lane_index: int) -> int:
        """変更先の車線の番号を取得
        Args:
            lane_index (int): 現在の車線の番号
        Returns:
            int: 車線変更先の車線 (上側: +1, 下側: -1)
        """
        if lane_index == 0:
            next_lane_index = +1
        elif lane_index == params.N_LANE-1:
            next_lane_index = -1
        else: # 目的車線はランダムに選択
            # if np.random.rand() < 0.5:
                # next_lane_index = +1
            # else:
                # next_lane_index = -1
            next_lane_index = -1
        return next_lane_index
    vehicle = agents[vehicle_index] # 車両
    # driving_modeを決定
    if lead_name == "obstacle": # 車線変更
        if vehicle.next_lane == None: # 変更先の車線を決定
            vehicle.next_lane = _decide_lane(lane_index=get_lane_index(pos_y=vehicle.pos[1]))
        # 目的車線の距離間隔を計算
        target_lane_index = get_lane_index(pos_y=vehicle.pos[1])+vehicle.next_lane # 目的車線の番号
        lead_gap = get_lead_gap(agents=agents, vehicle_index=vehicle_index, lane_index=target_lane_index)
        lag_gap = get_lag_gap(agents=agents, vehicle_index=vehicle_index, lane_index=target_lane_index)
        if judge_lane_change(vehicle=vehicle, lead_gap=lead_gap, lag_gap=lag_gap): # 車線変更の開始
            driving_mode = "lane_change"
            vehicle.lane_change_start_pos = vehicle.pos[0]
        else:
            driving_mode = "following"
    else: # 車両追従
        driving_mode = "following"
    return driving_mode

def reset_lane_change_variable(vehicle: object) -> None:
    """車線変更のパラメータを初期化
    Args:
        vehicle (object): _description_
    """
    vehicle.init_flag = True
    vehicle.driving_mode = "following"
    vehicle.cx = None
    vehicle.cy = None
    vehicle.next_lane = None
    vehicle.target_cource = None
    vehicle.target_lane_index = None
    vehicle.yaw = 0.0

def adjust_control_input(vehicle: object, accel: float, delta: float) -> tuple[float, float]:
    """制御入力を調整
    Args:
        vehicle (object): 追従車両エージェント
        accel (float): [m/s/s] 加速度
        delta (float): [rad] ステアリング角(車体に対する前輪タイヤの角度)
    Returns:
        tuple[float, float]: accel, delta
    """
    dt = params.DT
    # 躍度の調整
    jerk = (accel - vehicle.accel)/dt
    if jerk > params.JERK_MAX:
        jerk = params.JERK_MAX
    elif jerk < params.JERK_MIN:
        jerk = params.JERK_MIN
    # 加速度の調整
    accel = vehicle.accel + jerk*dt
    if accel > params.A_MAX:
        accel = params.A_MAX
    elif accel < params.A_MIN:
        accel = params.A_MIN
    # 速度の調整
    velocity = vehicle.velocity + accel*dt
    velocity = min(velocity, vehicle.max_vel)
    velocity = max(velocity, params.V_MIN)
    accel = (velocity - vehicle.velocity)/dt
    # ステアリング角の調整
    if delta > params.DELTA_MAX:
        delta = params.DELTA_MAX
    elif delta < params.DELTA_MIN:
        delta = params.DELTA_MIN
    return accel, delta

def get_control_input(lead_name: str, agents: list, vehicle_index: int) -> tuple[float, float]:
    """制御入力(accel, delta)を更新
    Args:
        lead_name (str): 前方オブジェクトの名前("obstacle" or "vehicle" or "none")
        agents (list): 車両エージェントのリスト
        vehicle_index (int): 車両の番号
    Returns:
        tuple[float, float]: accel (float): 
                             delta (float): 
    """
    # 初期設定 
    vehicle = agents[vehicle_index] # 車両
    lane_change_dis = params.LANE_RATE # [m] 車線変更をする距離
    if vehicle.driving_mode == None: # 初期設定
        vehicle.driving_mode = "following"
    # 走行モードを更新
    if vehicle.driving_mode == "following": # 車両追従
        vehicle.driving_mode = judge_driving_mode(lead_name=lead_name, agents=agents, vehicle_index=vehicle_index)
    else: # 車線変更 "lane_change"
        switch_dis = lane_change_dis + 30.0 # 車線変更を開始してから終了するまでの距離
        if vehicle.pos[0] - vehicle.lane_change_start_pos >= switch_dis:
            reset_lane_change_variable(vehicle)
            lane_index = get_lane_index(pos_y=vehicle.pos[1])
            vehicle.pos[1] = 1.5 + lane_index*params.W_LANE

    # 走行モードごとのaccel, deltaを取得
    if vehicle.driving_mode == "following": # 車両追従
        lead_dis = get_lead_gap(agents=agents, vehicle_index=vehicle_index, lane_index=get_lane_index(pos_y=vehicle.pos[1]))
        accel = optimal_velocity_model(vehicle=vehicle, lead_dis=lead_dis, pre_velocity=vehicle.velocity)
        delta = 0.0
    else: # 車線変更
        change_pos = [lane_change_dis, vehicle.next_lane*params.W_LANE]
        if vehicle.init_flag:
            step = vehicle.max_vel*params.DT/2
            vehicle.cx, vehicle.cy = car_agent.get_lanechange_path(init_pos=vehicle.pos, change_pos=change_pos, path_x=60, step=step)
            vehicle.target_course = car_agent.TargetCourse(vehicle.cx, vehicle.cy)
            vehicle.target_lane_index = get_lane_index(pos_y=vehicle.pos[1]) + vehicle.next_lane # 目的車線の番号 
            vehicle.init_flag = False
        target_ind, _ = vehicle.target_course.search_target_index(vehicle)
        delta, target_ind = car_agent.pure_pursuit_steer_control(vehicle, vehicle.target_course, target_ind)
        # target_lane_index = get_lane_index(pos_y=vehicle.pos[1]) + vehicle.next_lane # 目的車線の番号
        lead_dis = get_lead_gap(agents=agents, vehicle_index=vehicle_index, lane_index=vehicle.target_lane_index) 
        accel = optimal_velocity_model(vehicle=vehicle, lead_dis=lead_dis, pre_velocity=vehicle.velocity)
    # 制御入力の調整
    accel, delta = adjust_control_input(vehicle=vehicle, accel=accel, delta=delta)
    return accel, delta
