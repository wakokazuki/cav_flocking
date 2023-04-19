import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
import os
import sys

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params
from settings import show
from agent import agent
from path import car_agent

def main(show_flag, max_vel):
    # ラベルの設定
    driving_mode = 'inter'
    # environment
    direc_num = params.INT_DIREC_NUM # 交差点の方向の数(固定)
    lane_num = params.N_LANE
    # 参照する経路の生成
    init_pos = params.I_POS
    final_pos = params.F_POS
    cx, cy = car_agent.get_inter_path(max_vel=max_vel, init_pos=init_pos, final_pos=final_pos)
    max_timestep = params.NT # [-] シミュレーションの最大タイムステップ数
    # 初期設定
    order = 0
    vehicle = agent.Agent(state="C", set_params=None, max_vel=max_vel, init_pos=init_pos)
    vehicle.pos[:] = init_pos
    vehicle.yaw = np.pi/2
    vehicle.rear_x = vehicle.pos[0] - ((params.WB / 2) * np.cos(vehicle.yaw))
    vehicle.rear_y = vehicle.pos[1] - ((params.WB / 2) * np.sin(vehicle.yaw))
    lastIndex = len(cx) - 1
    timestep = 0
    statelist = agent.StateList(agent_num=1)
    statelist._append(num=order, vehicle=vehicle)
    target_course = car_agent.TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(vehicle)
    # シミュレーション
    if show_flag:
        fig, ax = plt.subplots(figsize=(6, 6))
        artists = []
        legend_flag = True

    while max_timestep >= timestep and lastIndex > target_ind:
        # 加速度を計算
        accel = car_agent.get_scalar_accel(desired_vel=max_vel, current_vel=vehicle.velocity)
        delta, target_ind = car_agent.pure_pursuit_steer_control(vehicle, target_course, target_ind)
        vehicle._update_kbm_rear(accel, delta)
        timestep += 1
        statelist._append(num=order, vehicle=vehicle)
        if show_flag:
            img = []
            if legend_flag:
                _center, _lane, _roadside = car_agent.intersection(inter_list=params.INT_LIST)
                for i in range(direc_num):
                    #center
                    ax.plot(_center[0,2*i:2*i+2], _center[1,2*i:2*i+2], color="y", ls="-", linewidth=2.0)
                    # lane
                    for la in range(lane_num-1):
                        _order = (lane_num-1)*4*i+4*la
                        ax.plot(_lane[0,_order:_order+2], _lane[1,_order:_order+2], color="k", ls="--")
                        ax.plot(_lane[0,_order+2:_order+4], _lane[1,_order+2:_order+4], color="k", ls="--")
                    # roadside
                    _order = 4*i
                    ax.plot(_roadside[0,_order:_order+2], _roadside[1,_order:_order+2], color="k", ls="-", linewidth=2.0)
                    ax.plot(_roadside[0,_order+2:_order+4], _roadside[1,_order+2:_order+4], color="k", ls="-", linewidth=2.0)
                ax.axis("equal")
                path_inter = mpatches.Arc(xy=(9.0, -9.0), width=27.0, height=27.0, theta1=90, theta2=180, color="r", linewidth=1.0)
                ax.add_patch(path_inter)
                path_enter = np.array([[-4.5, -29.0], [-4.5, -9.0]]).T
                ax.plot(path_enter[0], path_enter[1], color="r", linewidth=1.0)
                path_exit = np.array([[9.0, 4.5], [29.0, 4.5]]).T
                ax.plot(path_exit[0], path_exit[1], color="r", linewidth=1.0)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                legend_flag = False
            pos_x_list = [statelist.pos[order][i][0] for i in range(len(statelist.pos[0]))]
            pos_y_list = [statelist.pos[order][i][1] for i in range(len(statelist.pos[0]))]
            img += ax.plot(pos_x_list, pos_y_list, color="tab:blue", linewidth=1.0)
            img += ax.plot(pos_x_list[-1], pos_y_list[-1], marker=".", color="tab:blue")
            img += ax.plot(cx[target_ind], cy[target_ind], marker="x", color="tab:green", label="target")
            plt.tight_layout()
            artists.append(img)
    
    if show_flag:
        timestep = np.arange(len(statelist.velocity[order]))
        # リーダーの時間ごとの速度推移
        show.leader_path_velocity(x_list=timestep, y_list=statelist.velocity[order], driving_mode=driving_mode)
        # リーダーの時間ごとの車体方位角(yaw angle)の推移
        show.leader_path_yaw_angle(x_list=timestep, y_list=statelist.yaw[order], driving_mode=driving_mode)
        # シミュレーション
        # show.leader_path_inter()
        ani = animation.ArtistAnimation(fig, artists, interval=100)
        ani.save(f"../out/leader/video/{driving_mode}_route_speed{int(max_vel):02}.mp4", writer="ffmpeg", dpi=200)
    
    # CSVファイルへの保存
    # 位置
    with open(f"../out/leader/csv/{driving_mode}_speed{int(max_vel):02}_path_position.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(pos_x_list)
        writer.writerow(pos_y_list)
    # 速度
    vel_list = np.array(statelist.velocity[order])
    ang_list = np.array(statelist.yaw[order])
    vel_x = vel_list*np.cos(ang_list)
    vel_y = vel_list*np.sin(ang_list)
    with open(f"../out/leader/csv/{driving_mode}_speed{int(max_vel):02}_path_velocity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(np.around(vel_x, decimals=3))
        writer.writerow(np.around(vel_y, decimals=3))
            
if __name__ == '__main__':
    main(show_flag=True, max_vel=params.V_MAX)