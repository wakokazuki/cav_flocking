import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import os

# root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from parameters import params
from agent import agent
from settings import show
import car_agent

def main(show_flag: bool, max_vel: float, direction: str) -> None:
    # ラベルの設定
    driving_mode = "lane_change"
    # 車線変更の設定
    if direction == "up":
        lane_width = params.W_LANE # [m] 車線の幅
    if direction == "down":
        lane_width = - params.W_LANE # [m] 車線の幅
    change_dis = params.LANE_RATE # [m] 車線変更をする距離
    #  参照する経路の決定
    step = max_vel*params.DT/2
    cx, cy = car_agent.get_lanechange_path(init_pos=[0.0, 0.0], change_pos=[change_dis, lane_width], path_x=40.0, step=step)
    # 初期設定
    order = 0
    vehicle = agent.Agent(state="C", set_params=None, max_vel=max_vel, init_pos=[0,0])
    last_index = len(cx) - 1
    timestep = 0
    statelist = agent.StateList(agent_num=1)
    statelist._append(num=order, vehicle=vehicle)
    target_course = car_agent.TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(vehicle)
    # シミュレーション
    if show_flag:
        fig, ax = plt.subplots(figsize=(6,2))
        artists = []
        legend_flag = True

    time_count = 1
    while change_dis + 60 >= vehicle.pos[0] and last_index > target_ind:
        time_count += 1
        # 加速度を計算
        accel = car_agent.get_scalar_accel(desired_vel=max_vel, current_vel=vehicle.velocity)
        delta, target_ind = car_agent.pure_pursuit_steer_control(vehicle, target_course, target_ind)
        accel, delta = car_agent.adjust_control_input(accel=accel, delta=delta)
        vehicle._update_kbm_rear(accel, delta)
        timestep += 1
        statelist._append(num=0, vehicle=vehicle)
        if show_flag:
            img = []
            # 凡例の表示
            if legend_flag:
                # ax.set_xlabel("x [m]")
                # ax.set_ylabel("y [m]")
                ax.plot(cx, cy, color="tab:red")
                ax.grid()
                legend_flag = False
            pos_x_list = [statelist.pos[order][i][0] for i in range(len(statelist.pos[0]))]
            pos_y_list = [statelist.pos[order][i][1] for i in range(len(statelist.pos[0]))]
            img += ax.plot(pos_x_list, pos_y_list, color="tab:blue", linewidth=1.0)
            img += ax.plot(pos_x_list[-1], pos_y_list[-1], marker=".", color="tab:blue")
            img += ax.plot(cx[target_ind], cy[target_ind], marker="x", color="tab:green", label="target")
            plt.grid()
            artists.append(img)
    

    if show_flag: 
        timestep = np.arange(len(statelist.velocity[order]))
        # リーダーの時間ごとの速度推移
        show.leader_path_velocity(x_list=timestep, y_list=statelist.velocity[order], driving_mode=driving_mode)
        # リーダーの時間ごとの車体方位角(yaw angle)の推移
        show.leader_path_yaw_angle(x_list=timestep, y_list=statelist.yaw[order], driving_mode=driving_mode)
        # シミュレーション
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
            

if __name__ == "__main__":
    direction = "down" # 車線変更の方向 "up" or "down"
    main(show_flag=True, max_vel=params.V_MAX, direction=direction)