"""多目的最適化の評価関数値を計算
"""

import numpy as np
import copy
import math

from parameters import params
from settings import calc

def count_corr_value(agents: object, env: object) -> int:
    """速度整列指標を計算
    output:
        vel_corr
            float:速度整列指標
    """
    corr_count = 0.0 # init
    adj = env.adj
    y_axis = 1 # y-axis order
    for age in range(len(agents)):
        if params.VIRTUAL_LEADER:
            if age == params.ORD_LEADER:
                continue
        vehicle = agents[age]
        cluster_num = 0 # agent number within cluster distance
        corr_out = 0 # fitness value
        theta = env.calc_theta(vehicle.pos)
        _, inverse_matrix = calc.get_rotate_matrix(ang=theta)
        for i in range(len(agents)):
            if params.VIRTUAL_LEADER:
                if age == i or i == params.ORD_LEADER:
                    continue
            else:
                if age == i:
                    continue
            pos_diff = vehicle.pos - agents[i].pos
            distance, _ = calc.get_adjusted_difference(y_axis=y_axis, adj=adj, inverse_matrix=inverse_matrix, difference=pos_diff)
            if distance <= vehicle.cls_dis:
                cluster_num += 1
                between_angle = (vehicle.yaw + vehicle.slip) - (agents[i].yaw + agents[i].slip) # [rad] 2つの車両の速度のなす角
                corr_out = np.cos(between_angle)
                # inner = np.dot(vehicle.velocity, agents[i].velocity) # 内積の値    
                # corr_out += inner/(np.linalg.norm(vehicle.velocity)*np.linalg.norm(agents[i].velocity))
        if cluster_num != 0:
            corr_out /= cluster_num
            corr_count += 1 - corr_out # maximize -> minimize
    return corr_count

def count_coll_value(agents, env):
    """車同士・車と路端との衝突回数をカウント
    output:
        coll_count(int)
            int: number of collision
    """
    coll_count = 0 # init
    adj = env.adj
    y_axis = 1 # y-axis order
    max_length = params.DG # [m] 車両の対角線の長さ
    # 車両同士の衝突判定
    for age in range(len(agents)):
        vehicle = agents[age]
        if vehicle.state == "V": # 仮想リーダーの場合
            continue
        for i in range(len(agents)):
            if age == i or agents[i].state == "V":
                continue
            distance = np.linalg.norm(vehicle.pos - agents[i].pos)
            if distance <= max_length:
                if params.IS_INTER: # 交差点の場合
                    print("Error. Collision calculation")
                else: # 直線の場合
                    longitudinal_dis = abs(vehicle.pos[0] - agents[i].pos[0])
                    lateral_dis = abs(vehicle.pos[1] - agents[i].pos[1])
                    if longitudinal_dis <= params.CL and lateral_dis <= params.CW:
                        # print("Collision")
                        coll_count += 1 
    coll_count /= 2
    # for age in range(len(agents)):
    #     if params.VIRTUAL_LEADER:
    #         if age == params.ORD_LEADER:
    #             continue
            

        
    #     theta = env.calc_theta(agents[age].pos)
    #     _, inverse_matrix = calc.get_rotate_matrix(ang=theta)
    #     for i in range(len(agents)):
    #         if params.VIRTUAL_LEADER:
    #             if age == i or i == params.ORD_LEADER:
    #                 continue
    #         else:
    #             if age == i:
    #                 continue
    #         pos_diff = agents[age].pos - agents[i].pos
    #         distance, _ = calc.get_adjusted_difference(y_axis=y_axis, adj=adj, inverse_matrix=inverse_matrix, difference=pos_diff)
    #         if distance < params.COLL_DIS:
    #             coll_count += 1
    # coll_count = coll_count/2
    # collision with road edge
    for age in range(len(agents)):
        if params.VIRTUAL_LEADER:
            if age == params.ORD_LEADER:
                continue
        road_pos = env.calc_lane(agents[age].pos)
        half_road = params.SIZE_Y/2 
        to_center = half_road - road_pos
        dis_from_wall = half_road - abs(to_center)
        if dis_from_wall < params.CW/2:
            coll_count += 1
    return coll_count

def count_obs_value(agents, env):
    """障害物・壁との衝突回数を計算
    output:
        obs_val
            int:障害物・壁との衝突回数
    """
    obs_count = 0 # init
    # calc collision with obstacle
    if params.IS_OBSTACLE:
        if params.IS_INTER:
            print("COUNT OBS ERROR")
        for keys in params.OBS_DICT.keys():
            if params.OBS_DICT[keys]:
                for age in range(len(agents)):
                    if params.VIRTUAL_LEADER:
                        if age == params.ORD_LEADER:
                            continue
                    longitudinal_dis = abs(agents[age].pos[0] - params.OBS_POS_DICT[keys][0])
                    lateral_dis = abs(agents[age].pos[1] - params.OBS_POS_DICT[keys][1])
                    # print(longitudinal_dis, lateral_dis)
                    if longitudinal_dis <= (params.RAD_OBS+params.CL) and lateral_dis <= (params.RAD_OBS+params.CW/2):
                        obs_count += 1
    return obs_count

def count_dis_value(agents, env):
    """車間距離の指標を計算
    """
    dis_count = 0.0 # init
    adj = env.adj
    y_axis = 1 # y-axis order
    for age in range(len(agents)):
        if params.VIRTUAL_LEADER:
            if age == params.ORD_LEADER:
                continue
        cluster_num = 0 # agent number within cluster distance
        output = 0 # init
        theta = env.calc_theta(agents[age].pos)
        _, inverse_matrix = calc.get_rotate_matrix(ang=theta)
        for i in range(len(agents)):
            if params.VIRTUAL_LEADER:
                if age == i or i == params.ORD_LEADER:
                    continue
            else:
                if age == i:
                    continue
            pos_diff = agents[age].pos - agents[i].pos
            distance, _ = calc.get_adjusted_difference(y_axis=y_axis, adj=adj, inverse_matrix=inverse_matrix, difference=pos_diff) 
            if distance <= agents[age].cls_dis:
                cluster_num += 1
                output += np.abs(distance - agents[age].des_dis)
        if cluster_num != 0:
            dis_count += output/cluster_num
        else: # no connectivity 
            # print("No Connection with Other Vehicle")
            dis_count += 20
    # dis_count /= params.N_FOLLOWER 
    dis_count /= env.n_follower 
    # print(dis_count)
    return dis_count

def eval_corr_value(corr, env):
    """Evaluate Correlation
    """
    total = params.NT*env.n_follower
    corr_value = corr/total
    return corr_value 

def eval_dis_value(dis, env):
    """evaluate distance between CAV
    """
    dis_value = dis/params.NT
    # print(dis_value)
    return dis_value

