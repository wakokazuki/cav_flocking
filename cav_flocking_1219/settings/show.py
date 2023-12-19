"""結果の画像と動画を出力"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, animation, gridspec

from parameters import params
from path import car_agent

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.family"] = "Meiryo"
# plt.rcParams['text.usetex'] = True

def agent_plot(img: plt, ax: plt, t: int, agents: list, pos_list: list, leader_pos: np.ndarray or None, leader_flag: bool) -> None:
    """各エージェントの位置座標をプロット
    """
    n_lead = params.N_LEADER
    for i in range(params.N_FOLLOWER):
        if agents[n_lead+i].state == "F":
            img += ax.plot(pos_list[i][t][0], pos_list[i][t][1], ".", markersize=20, color="tab:red", zorder=2)
        elif agents[n_lead+i].state == "C":
            img += ax.plot(pos_list[i][t][0], pos_list[i][t][1], ".", markersize=20, color="tab:blue", zorder=2)
    if leader_flag == True:
        if params.VIRTUAL_LEADER:
            img += ax.plot(leader_pos[0, t], leader_pos[1, t], "s", markersize=12, alpha=0.5, color=params.COLORS[0], zorder=2) 
        else:   
            img += ax.plot(leader_pos[0, t], leader_pos[1, t], "s", markersize=12, color=params.COLORS[0]) 
    return img

def save_frame(agents: list, pos_list: list, leader_pos: np.ndarray or None, leader_flag: bool) -> None:
    """各タイムステップのエージェントの位置座標をプロット"""
    for t in params.T_LIST:
        fig, ax = plt.subplots(figsize=(8.5, 3))
        artist = []
        artist += agent_plot(img=artist, ax=ax, t=t, agents=agents, pos_list=pos_list, leader_pos=leader_pos, leader_flag=leader_flag)
        ax.set_xlim(-50, params.SIZE_X)
        ax.set_ylim(0, params.SIZE_Y)
        ax.set_yticks([params.W_LANE*i for i in range(params.N_LANE+1)])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        for i in range(params.N_LANE-1): # Road Lane
            ax.axhline(params.W_LANE*(i+1), ls="--", zorder=2, color="black")
        if params.IS_OBSTACLE:
            for keys in params.OBS_DICT.keys():
                if params.OBS_DICT[keys]:
                    # 実際の大きさ
                    # circle = patches.Circle(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), radius=params.RAD_OBS, facecolor="black", zorder=2)
                    # ax.add_patch(circle)
                    # スライド用に大きさ調整
                    ellipse = patches.Ellipse(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), width=params.RAD_OBS*4, height=params.RAD_OBS*2, facecolor="black", zorder=2) # スライド用
                    ax.add_patch(ellipse)
        for fol in range(params.N_FOLLOWER):
            x_list = [pos_list[fol][i][0] for i in range(len(pos_list[fol]))]
            y_list = [pos_list[fol][i][1] for i in range(len(pos_list[fol]))]
            plt.plot(x_list, y_list, color=params.F_COLORS[fol], linewidth=2.0, alpha=0.5, zorder=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"../out/img/frame/time{t:03}.png", dpi=150)
        plt.clf()
        plt.close()

    """最終位置をプロット
    """
    fig, ax = plt.subplots(figsize=(8.5, 3))
    artist = []
    artist += agent_plot(img=artist, ax=ax, t=0, agents=agents, pos_list=pos_list, leader_pos=leader_pos, leader_flag=leader_flag)
    artist += agent_plot(img=artist, ax=ax, t=t, agents=agents, pos_list=pos_list, leader_pos=leader_pos, leader_flag=leader_flag)
    ax.set_xlim(-50, params.SIZE_X)
    ax.set_ylim(0, params.SIZE_Y)
    ax.set_yticks([params.W_LANE*i for i in range(params.N_LANE+1)])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xticks([-50,0,50,100,150,200,250,300])
    ax.set_yticks([0,3,6,9])
    for i in range(params.N_LANE-1): # Road Lane
        ax.axhline(params.W_LANE*(i+1), ls="--", zorder=2, color="black")
    if params.IS_OBSTACLE:
        for keys in params.OBS_DICT.keys():
            if params.OBS_DICT[keys]:
                # 実際の大きさ
                # circle = patches.Circle(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), radius=params.RAD_OBS, facecolor="black", zorder=2)
                # ax.add_patch(circle)
                # スライド用に大きさ調整
                ellipse = patches.Ellipse(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), width=params.RAD_OBS*4, height=params.RAD_OBS*2, facecolor="black", zorder=2) # スライド用
                ax.add_patch(ellipse)
    for fol in range(params.N_FOLLOWER):
        x_list = [pos_list[fol][i][0] for i in range(len(pos_list[fol]))]
        y_list = [pos_list[fol][i][1] for i in range(len(pos_list[fol]))]
        plt.plot(x_list, y_list, color=params.F_COLORS[fol], linewidth=2.0, alpha=0.75, zorder=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/frame/path_init_last.png", dpi=150)
    plt.clf()
    plt.close()

    

def animation_straight(file_name: str, date_time: str, agents: list, pos_list: list, leader_pos: np.ndarray or None, leader_flag: bool) -> None:
    """直線の道路環境のアニメーションを作成
    Args:
        file_name (str): ファイル名
        date_time (str): 作成した時間
        agents (list): エージェントのリスト
        pos_list (list): 位置のリスト
        leader_pos (np.ndarray or None): リーダーの位置(リーダーがいない場合, None)
        leader_flag (bool): リーダーがいる場合, True
    """
    fig, ax = plt.subplots(figsize=(8.5, 3))
    artists = []
    legend_flag = True # 凡例の描写
    for t in range(params.NT):
        artist = []
        artist += agent_plot(img=artist, ax=ax, t=t, agents=agents, pos_list=pos_list, leader_pos=leader_pos, leader_flag=leader_flag) 
        if legend_flag:
            ax.set_xlim(-50, params.SIZE_X)
            ax.set_ylim(0, params.SIZE_Y)
            ax.tick_params(labelbottom=True)
            ax.tick_params(bottom=True)
            ax.tick_params(labelleft=True)
            ax.tick_params(left=True)
            ax.set_yticks([params.W_LANE*i for i in range(params.N_LANE+1)])
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            for i in range(params.N_LANE-1): # Road Lane
                ax.axhline(params.W_LANE*(i+1), ls="--", zorder=1, color="black")
            if params.IS_OBSTACLE:
                for keys in params.OBS_DICT.keys():
                    if params.OBS_DICT[keys]:
                        # 実際の大きさ
                        # circle = patches.Circle(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), radius=params.RAD_OBS, facecolor="black", zorder=2)
                        # ax.add_patch(circle)
                        # スライド用に大きさ調整
                        ellipse = patches.Ellipse(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), width=params.RAD_OBS*4, height=params.RAD_OBS*2, facecolor="black", zorder=2) # スライド用
                        ax.add_patch(ellipse)
            legend_flag = False
            # 初期位置
            plt.tight_layout()
            plt.savefig(f"../out/img/init/road_num{params.N_FOLLOWER}_speed{int(params.V_MAX):02}.png", dpi=150)
        artists.append(artist)
    ani = animation.ArtistAnimation(fig, artists, interval=50)
    ani.save(f"../out/video/{file_name}_straight_{date_time}.mp4", writer="ffmpeg", dpi=200)

    # 途中の位置座標を表示
    save_frame(agents=agents, pos_list=pos_list, leader_pos=leader_pos, leader_flag=leader_flag)

def animation_inter(file_name: str, date_time: str, agents: list, pos_list: list, leader_pos: np.ndarray or None, leader_flag: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    artists = []
    legend_flag = True # 凡例の描写
    for t in range(params.NT):
        artist = []
        artist += agent_plot(img=artist, ax=ax, t=t, agents=agents, pos_list=pos_list, leader_pos=leader_pos, leader_flag=leader_flag) 
        if legend_flag:
            direc_num = params.INT_DIREC_NUM # 交差点の方向の数(固定)
            lane_num = 3
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
                # inter 
                theta_circle =  np.linspace(np.pi/2, np.pi, 50)
                for i in range(params.N_LANE + 1):
                    radius = 9 + 3 * i
                    x = radius * np.cos(theta_circle) +9
                    y = radius * np.sin(theta_circle) -9
                    ax.plot(x, y,color = 'black')
            ax.axis("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            legend_flag = False
            # 初期位置
            plt.tight_layout()
            plt.savefig(f"../out/img/init/inter_num{params.N_FOLLOWER}_speed{int(params.V_MAX):02}.png", dpi=150)
        plt.tight_layout()
        artists.append(artist)
    ani = animation.ArtistAnimation(fig, artists, interval=50)
    ani.save(f"../out/video/{file_name}_inter_{date_time}.mp4", writer="ffmpeg", dpi=150)

def velocity(file_name: str, x_list: list, y_list: list) -> None:
    plt.figure(figsize=(6, 3))
    for fol in range(params.N_FOLLOWER):
        plt.plot(x_list, y_list[fol], color=params.F_COLORS[fol], zorder=3)
    plt.axhline(y=params.V_MIN, ls="-", color="black", zorder=2)
    plt.axhline(y=params.V_MAX, ls="-", color="black", zorder=2)
    plt.ylim(params.V_MIN-0.5, params.V_MAX+0.5)
    plt.yticks([0,2,4,6.0,8])
    # plt.xlabel("elapsed time (s)")
    # plt.ylabel("velocity (m/s)")
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/velocity/vel_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

def acceleration(file_name: str, x_list: list, y_list: list) -> None:
    plt.figure(figsize=(6, 3))
    for fol in range(params.N_FOLLOWER):
        plt.plot(x_list, y_list[fol], color=params.F_COLORS[fol], zorder=3)
    plt.axhline(y=params.A_MIN, ls="-", color="black", zorder=2)
    plt.axhline(y=params.A_MAX, ls="-", color="black", zorder=2)
    plt.ylim(params.A_MIN-0.5, params.A_MAX+0.5)
    # plt.xlabel("elapsed time (s)")
    # plt.ylabel(r"acceleration ($\mathrm{m/s^2}$)")
    plt.xlabel("時間 (s)")
    plt.ylabel(r"加速度 ($\mathrm{m/s^2}$)")
    plt.yticks([-4, -2, 0, 2])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/accel/accel_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

def heading_angle(file_name: str, x_list: list, y_list:list) -> None:
    plt.figure(figsize=(6, 3))
    for fol in range(params.N_FOLLOWER):
        plt.plot(x_list, y_list[fol], color=params.F_COLORS[fol], zorder=3)
    plt.yticks([-np.pi/8, 0, np.pi/8],[r"$\frac{-\pi}{8}$", 0, r"$\frac{\pi}{8}$"])
    plt.xlabel("elapsed time (s)")
    plt.ylabel("yaw angle (rad)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/angle/yaw_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

    plt.yticks([-4, -2, 0, 2])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/accel/accel_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

def heading_angle(file_name: str, x_list: list, y_list:list) -> None:
    plt.figure(figsize=(6, 3))
    for fol in range(params.N_FOLLOWER):
        plt.plot(x_list, y_list[fol], color=params.F_COLORS[fol], zorder=3)
    plt.yticks([-np.pi/8, 0, np.pi/8],[r"$\frac{-\pi}{8}$", 0, r"$\frac{\pi}{8}$"])
    plt.xlabel("elapsed time (s)")
    plt.ylabel("yaw angle (rad)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/angle/yaw_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

def steering_angle(file_name: str, x_list: list, y_list:list) -> None:
    plt.figure(figsize=(6, 3))
    for fol in range(params.N_FOLLOWER):
        plt.plot(x_list, y_list[fol], color=params.F_COLORS[fol], zorder=3)
    plt.axhline(y=params.DELTA_MIN, ls="-", color="black", zorder=2)
    plt.axhline(y=params.DELTA_MAX, ls="-", color="black", zorder=2)
    plt.yticks([params.DELTA_MIN, 0, params.DELTA_MAX],[r"$\frac{-\pi}{16}$", 0, r"$\frac{\pi}{16}$"])
    plt.xlabel("elapsed time (s)")
    plt.ylabel("steering angle (rad)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/angle/steering_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()


def straight_path_trajectory(file_name: str, pos_list: list) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3))
    plt.xlim(-50, params.SIZE_X)
    plt.ylim(0, params.SIZE_Y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.yticks([params.W_LANE*i for i in range(params.N_LANE+1)])
    for i in range(params.N_LANE-1): # Road Lane
        ax.axhline(params.W_LANE*(i+1), ls="--", zorder=2, color="black")
    if params.IS_OBSTACLE:
        for keys in params.OBS_DICT.keys():
            if params.OBS_DICT[keys]:
                # 実際の大きさ
                # circle = patches.Circle(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), radius=params.RAD_OBS, facecolor="black", zorder=2)
                # ax.add_patch(circle)
                # スライド用に大きさ調整
                ellipse = patches.Ellipse(xy=(params.OBS_POS_DICT[keys][0], params.OBS_POS_DICT[keys][1]), width=params.RAD_OBS*4, height=params.RAD_OBS*2, facecolor="black", zorder=2) # スライド用
                ax.add_patch(ellipse) 
    for fol in range(params.N_FOLLOWER):
        x_list = [pos_list[fol][i][0] for i in range(len(pos_list[fol]))]
        y_list = [pos_list[fol][i][1] for i in range(len(pos_list[fol]))]
        plt.plot(x_list, y_list, color=params.F_COLORS[fol], linewidth=2.0)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/route/route_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

def inter_path_trajectory(file_name: str, pos_list: list) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    direc_num = params.INT_DIREC_NUM
    lane_num = 3
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
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for fol in range(params.N_FOLLOWER):
        x_list = [pos_list[fol][i][0] for i in range(len(pos_list[fol]))]
        y_list = [pos_list[fol][i][1] for i in range(len(pos_list[fol]))]
        plt.plot(x_list, y_list, color=params.F_COLORS[fol], linewidth=2.0)
    #カーブ部分の車線表示
    theta_circle =  np.linspace(np.pi/2, np.pi, 50)
    for i in range(params.N_LANE + 1):
        radius = 9 + 3 * i
        x = radius * np.cos(theta_circle) +9
        y = radius * np.sin(theta_circle) -9
        ax.plot(x, y,color = 'black')
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/img/route/route_{file_name}.png", dpi=150)
    plt.clf()
    plt.close()

def leader_path_velocity(x_list: list, y_list: list, driving_mode: str) -> None:
    plt.figure(figsize=(6,3))
    plt.plot(x_list, y_list, color="tab:red")
    plt.xlabel(r"Elapsed Time ($s$)")
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/leader/img/{driving_mode}_speed.png", dpi=150)
    plt.clf()
    plt.close()

def leader_path_yaw_angle(x_list: list, y_list: list, driving_mode: str) -> None:
    plt.figure(figsize=(6,3))
    plt.plot(x_list, y_list, color="tab:red")
    plt.xlabel("Timestep")
    plt.ylabel("Yaw angle [rad]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/leader/img/{driving_mode}_yaw_angle.png", dpi=150)
    plt.clf()
    plt.close()

def objective_value_convergence(value: np.ndarray, date_time: str, loop_num: int, method_label: str) -> None:
    """最適化問題の世代ごとの目的関数値の収束性を表示

    Args:
        value (np.ndarray): 目的関数値のリスト
        date_time (str): 実行時間 (ファイル名)
        loop_num (int): 反復ステップ数
        method_label (str): 手法 (ファイル名)
    """
    x = np.arange(1, params.N_GEN+1)
    fig, ax = plt.subplots(figsize=(6,4))
    plt.plot(x, value, "-")
    plt.xlabel("Number of generations")
    plt.ylabel("Objective value")
    plt.title("Parameter convergence")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../out/convergence/img/{date_time}_loop{loop_num}_pop{params.N_POP}_gen_{params.N_GEN}_age{int(params.N_FOLLOWER):02}_vel{int(params.V_MAX):02}_{method_label}.png", dpi=150)
    plt.clf()
    plt.close()
