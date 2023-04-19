import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from parameters import params
from settings import show
from agent import agent
from path import car_agent

def main(show_flag, max_vel):
    # ラベルの設定
    driving_mode = "straight"
    # 参照する経路の生成
    cx, cy = car_agent.get_straight_path(x=params.L_POS[0], y=params.L_POS[1], nt=params.NT, velocity=max_vel)
    max_timestep = params.NT # [-] シミュレーションの最大タイムステップ数
    # 初期設定
    order = 0
    vehicle = agent.Agent(state="C", set_params=None, max_vel=max_vel, init_pos=params.L_POS)
    vehicle.pos[:] = params.L_POS
    vehicle.rear_x = vehicle.pos[0] - ((params.WB / 2) * np.cos(vehicle.yaw))
    vehicle.rear_y = vehicle.pos[1] - ((params.WB / 2) * np.sin(vehicle.yaw))
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

    while max_timestep >= timestep and last_index > target_ind:
        # 加速度を計算
        accel = car_agent.get_scalar_accel(desired_vel=max_vel, current_vel=vehicle.velocity)
        delta, target_ind = car_agent.pure_pursuit_steer_control(vehicle, target_course, target_ind)
        # print(delta)
        vehicle._update_kbm_rear(accel, delta)
        timestep += 1
        statelist._append(num=0, vehicle=vehicle)
        pos_x_list = [statelist.pos[order][i][0] for i in range(len(statelist.pos[0]))]
        pos_y_list = [statelist.pos[order][i][1] for i in range(len(statelist.pos[0]))]
        if show_flag:
            img = []
            # 凡例の表示
            if legend_flag:
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.plot(cx, cy, color="tab:red")
                legend_flag = False
            img += ax.plot(pos_x_list, pos_y_list, color="tab:blue", linewidth=1.0)
            img += ax.plot(pos_x_list[-1], pos_y_list[-1], marker=".", color="tab:blue")
            img += ax.plot(cx[target_ind], cy[target_ind], marker="x", color="tab:green", label="target")
            plt.grid()
            artists.append(img)

    if show_flag: 
        timestep = np.arange(len(statelist.velocity[order]))
        # リーダーの時間ごとの速度推移
        show.leader_path_velocity(x_list=timestep, y_list=statelist.velocity[0], driving_mode=driving_mode)
        # リーダーの時間ごとの車体方位角(yaw angle)の推移
        show.leader_path_yaw_angle(x_list=timestep, y_list=statelist.yaw[0], driving_mode=driving_mode)
        # シミュレーション
        ani = animation.ArtistAnimation(fig, artists, interval=100)
        ani.save(f"../out/leader/video/{driving_mode}_route.mp4", writer="ffmpeg", dpi=200)
    
    # CSVファイルへの保存
    # 位置
    with open(f"out/leader/csv/{driving_mode}_speed{int(max_vel):02}_path_position.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(pos_x_list)
        writer.writerow(pos_y_list)
    # 速度
    vel_list = np.array(statelist.velocity[order])
    ang_list = np.array(statelist.yaw[order])
    vel_x = vel_list*np.cos(ang_list)
    vel_y = vel_list*np.sin(ang_list)
    with open(f"out/leader/csv/{driving_mode}_speed{int(max_vel):02}_path_velocity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(np.around(vel_x, decimals=3))
        writer.writerow(np.around(vel_y, decimals=3))

# if __name__ == "__main__":
#     main(show_flag=True, max_vel=params.V_MAX)
            