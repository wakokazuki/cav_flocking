"""プログラム上の計算を実行"""

import numpy as np
import math
import sys
import os

# root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from parameters import params

def get_obstacle_zone(obs_pos: list) -> list:
    """障害物の影響範囲(obstacle_zone)を取得
    Args:
        obs_pos (list): 障害物の位置座標 [x,y]
    Returns:
        list: 障害物の影響範囲 [[x_min, y_min],[x_max, y_max]]
    """
    # 初期設定
    zone_min = []
    zone_max = []
    # x座標
    zone_min.append(obs_pos[0]-params.OBS_NORTICE_DIS)
    zone_max.append(obs_pos[0])
    # y座標
    for lan in range(params.N_LANE):
        if lan*params.W_LANE <= obs_pos[1] <= (lan+1)*params.W_LANE:
            lane_min = lan*params.W_LANE
        else:
            pass
    lane_max = lane_min + params.W_LANE
    lane_min = max(obs_pos[1]-params.W_LANE, 0)
    lane_max = min(obs_pos[1]+params.W_LANE, params.SIZE_Y)
    zone_min.append(lane_min)
    zone_max.append(lane_max)
    return [zone_min, zone_max]

def judge_obstacle_zone(agent_pos: list, obs_zone: list) -> bool:
    """エージェントが障害物の影響範囲内か判定
    Args:
        agent_pos (list): エージェントの位置座標 [x,y]
        obs_zone (list): 障害物の影響範囲 [[x_min, y_min],[x_max, y_max]] 
    Returns:
        bool: 障害物の影響範囲内かどうか
    """
    # x-axis, y-axis
    if obs_zone[0][0] <= agent_pos[0] <= obs_zone[1][0] and obs_zone[0][1] <= agent_pos[1] <= obs_zone[1][1]:
        is_zone = True
    else:
        is_zone = False
    return is_zone  

def adjust_obstacle_parameter(pos: np.ndarray, rad: float):
    # 位置の調整
    if pos[1] <= params.W_LANE:
        pos[1] = 0.0
    elif pos[1] >= params.SIZE_Y-params.W_LANE:
        pos[1] = params.SIZE_Y
    else:
        pass
    # 半径の調整
    if pos[1] <= params.W_LANE or pos[1] >= params.SIZE_Y-params.W_LANE:
        rad *= 2
    else:
        pass
    return pos, rad

def get_rotate_matrix(ang: float) -> tuple[np.ndarray, np.ndarray]:
    """回転行列を取得
    Args:
        ang (float): [rad] 回転角
    Returns:
        tuple[np.ndarray, np.ndarray]: 回転行列, 逆行列
    """
    rotate_matrix = np.array([[np.cos(ang), -np.sin(ang)],
                              [np.sin(ang), np.cos(ang)]]) # 回転行列
    inverse_matrix = np.array([[np.cos(-ang), -np.sin(-ang)],
                               [np.sin(-ang), np.cos(-ang)]]) # 　逆行列
    return rotate_matrix, inverse_matrix

def get_adjusted_difference(y_axis: int, adj: float, inverse_matrix: list, difference: list):
    """get adjusted difference(distance, velocity) and rotate difference [x,y]

    Args:
        y_axis (int): order of y-axis
        adj (float): adjustment ratio
        inverse_matrix (list): 
        difference (list): distance or velocity
    """
    rotate_difference = np.dot(inverse_matrix, difference.T)
    # adjust longitudinal-lateral ratio
    rotate_difference[y_axis] *= adj
    adjusted_difference = np.linalg.norm(rotate_difference)
    return adjusted_difference, rotate_difference

def get_braking_distance(velocity: float, veh_dis: float) -> float:
    """Flockingアルゴリズムの希望車間距離を計算
    Args:
        velocity (float): [m/s] リーダーの速度
        veh_dis (float): [m] 停止時の車間距離
    Returns:
        float: [m] 希望車間距離 = 制動距離 + 停止時の車間距離
    """
    frict = 0.7 # 摩擦係数(乾いた路面:0.7, 濡れた路面:0.5)
    hour_vel = velocity*3600/1000 # [h/km] 速度の時速換算
    # 制動距離 = 時速の2乗 / (254 * 摩擦係数)
    dis = hour_vel**2/(254*frict) + veh_dis # [m] 制動距離
    return dis

def get_longitudinal_vec(ang, vector):
    y_axis = 1
    rotate_matrix, inverse_matrix = get_rotate_matrix(ang=ang)
    rotate_vec = np.dot(inverse_matrix, vector.T)
    rotate_vec[y_axis] = 0.0
    longitudinal_vec = np.dot(rotate_matrix, rotate_vec.T)
    return longitudinal_vec

def get_velocity_within_limits(pre_velocity, pre_angle, des_accel, max_angle, max_velocity, max_accel):
    """get velocity within limits (max angle, max accel, max velocity)
    """
    velocity = pre_velocity + des_accel*params.DT
    # theta = np.arctan2(pre_velocity[1], pre_velocity[0])
    theta = pre_angle
    # print(f"theta:{theta}")
    rotate_matrix, inverse_matrix = get_rotate_matrix(ang=theta)
    rotate_velocity = np.dot(inverse_matrix, velocity)
    # print(f"rotate_velocity:{rotate_velocity}")
    rotate_theta = np.arctan2(rotate_velocity[1], rotate_velocity[0])
    # print(f"rotate_theta:{rotate_theta}")
    # steering angle and accel limit
    # print(f"np.pi - abs(max_angle):{np.pi - abs(max_angle)}")
    if abs(rotate_theta) <= abs(max_angle) or abs(rotate_theta) >= (np.pi - abs(max_angle)):
        angle = rotate_theta
        accel_value = min(max_accel, np.linalg.norm(des_accel))
        # print(f"des_accel:{des_accel}")
        regulate_accel = accel_value*np.dot(inverse_matrix, des_accel)/np.linalg.norm(des_accel)
        # print(f"accel:{accel_value}")
        # print("good")
    elif rotate_theta < -max_angle:
        angle = -max_angle
        ang_diff = -max_angle - rotate_theta
        # print(f"ang_diff:{ang_diff}")
        unit_vec = np.array([np.cos(-max_angle), np.sin(-max_angle)])
        vel_within_ang = velocity*np.cos(ang_diff)*unit_vec
        changed_accel = (vel_within_ang - pre_velocity)/params.DT
        accel_value = min(max_accel, np.linalg.norm(changed_accel))
        regulate_accel = accel_value*unit_vec/np.linalg.norm(unit_vec)
        # print(regulate_accel)
        # print("down")

    elif rotate_theta > max_angle:
        angle = max_angle
        ang_diff = rotate_theta - max_angle
        # print(f"ang_diff:{ang_diff}")
        unit_vec = np.array([np.cos(max_angle), np.sin(max_angle)])
        # print(f"unit_vec:{unit_vec}")
        vel_within_ang = velocity*np.cos(ang_diff)*unit_vec
        changed_accel = (vel_within_ang - pre_velocity)/params.DT
        accel_value = min(max_accel, np.linalg.norm(changed_accel))
        # print(f"accel_value:{accel_value}")
        regulate_accel = accel_value*unit_vec/np.linalg.norm(unit_vec)
        # print("up")
    # print(f"regulate_accel:{regulate_accel}")

    # velocity limit
    regulate_rotate_velocity = np.dot(inverse_matrix, pre_velocity) + regulate_accel*params.DT
    # print(f"regulate_rotate_velocity:{regulate_rotate_velocity}")
    velocity = np.dot(rotate_matrix, regulate_rotate_velocity)
    # print(f"velocity:{velocity}") 2
    if np.linalg.norm(velocity) >= max_velocity:
        # print("max_vel")
        velocity = max_velocity*velocity/np.linalg.norm(velocity)
    accel = np.dot(rotate_matrix, regulate_accel)
    return velocity, accel, angle


    