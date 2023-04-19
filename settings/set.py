import numpy as np

from parameters import params
from settings import calc

def calc_flock(agents, env, t) -> None:
    """次時刻の状態を計算"""
    for i in range(len(agents)):
        agents[i]._calc_next(i, agents, env, t)
    # 状態を更新
    for age in range(len(agents)):
        agents[age]._update_state() 

def get_parameter_order(param_dict, param_order) -> int:
    order_num = 0 # init
    for keys in params.LABEL_DICT.keys():
        if keys == param_dict:
          break
        order_num += params.FRC_BIN_DICT[keys]*len(params.LABEL_DICT[keys])
    order_num += param_order
    return order_num

def get_init_pos_straight(order: int, des_dis: float) -> list:
    """片側3車線の直進道路の場合の, Flocking車両の初期位置を計算
    Args:
        order (int): [-] Flocking車両の順番
        des_dis (float): [m] 希望車間距離
    Returns:
        list: [x,y] Flocking車両の初期位置
    """
    leader_pos = params.L_POS
    q, mod = divmod(order, params.N_LANE) # q:商, mod:余り
    if params.N_LANE == 2:
        if mod == 0:
            pos_list = [leader_pos[0]-(q+0.5)*des_dis, 1.5]
        elif mod == 1: 
            pos_list = [leader_pos[0]-(q+1)*des_dis, 4.5]
    elif params.N_LANE == 3:
        if mod == 0:
            pos_list = [leader_pos[0]-(q+0.5)*des_dis, 1.5]
        elif mod == 1:
            pos_list = [leader_pos[0]-(q+0.5)*des_dis, 7.5]
        else: # mod == 2
            pos_list = [leader_pos[0]-(q+1)*des_dis, 4.5]
    return pos_list 

def get_init_pos_straight_random(des_dis: float) -> list:
    pos_x = -np.random.rand()*des_dis*(params.N_FOLLOWER//params.N_LANE)
    pos_y = np.random.rand()*(params.N_LANE-1)*params.W_LANE + params.W_LANE/2
    pos_list = [pos_x, pos_y]
    return pos_list

def get_init_pos_inter(order: int,  des_dis: float) -> list:
    leader_pos = params.I_POS
    q, mod = divmod(order, params.N_LANE) # q:商, mod:余り
    if mod == 0:
        pos_list = [-1.5, leader_pos[0]-(q+0.5)*des_dis]
    elif mod == 1:
        pos_list = [-7.5, leader_pos[0]-(q+0.5)*des_dis]
    else: # mod == 2
        pos_list = [-4.5, leader_pos[0]-(q+1)*des_dis]
    return pos_list 

def get_init_pos_inter_random(seed: int, agent_num: int) -> list:
    """交差点のランダムな初期位置のリストを出力
    Args:
        seed (int): シード値
        agent_num (int): エージェント数
    Returns:
        list: ランダムな初期位置のリスト 
    """
    init_pos_list = []
    direc_num = 4
    lane_num = params.N_LANE
    np.random.seed(seed=seed)
    for i in range(agent_num):
        direc_ord = np.random.randint(1, direc_num+1)
        lane_ord = np.random.randint(1, lane_num+1)
        pos = np.array([params.W_LANE*lane_num+params.INT_LIST[direc_ord-1], -params.W_LANE*(lane_ord-0.5)])
        rotate_matrix, _ = calc.get_rotate_matrix(ang=(direc_ord-1)*np.pi/2)
        init_pos = np.around(np.dot(rotate_matrix, pos), decimals=3)
        init_pos_list.append(init_pos.tolist())
    return init_pos_list


def get_flock_parameter(set_params: list, frc_key: str) -> list:
    """flockingのパラメータを取得
    Args:
        set_params (list): Flockingパラメータのリスト
        key (str): Flockingのルール(FRC)のkey
    Returns:
        list: frc_param_list: Flockingアルゴリズムのうちの, 1つのルールのパラメータのリスト
    """
    if params.FRC_DICT[frc_key]:
        count = 0
        param_num = len(params.LABEL_DICT[frc_key])
        for keys in params.FRC_DICT.keys():
            if frc_key == keys:
                frc_param_list = set_params[count: count+param_num]
                return frc_param_list
            elif params.FRC_DICT[keys]:
               count += len(params.LABEL_DICT[keys])
        return None # Error用
    else:
        return None # Error用

def set_parameter_limit(lower, upper, frc_dict):
    """set parameter limit (for pymoo)
    """
    # init
    xl = [] # lower
    xu = [] # upper
    for keys in frc_dict.keys():
        if frc_dict[keys]:
            xl.extend(lower[keys])
            xu.extend(upper[keys])
    return xl, xu

def set_column(frc_dict, label_dict, obj_dict, obj_label_dict, single_obj=False):
    """set parameter column (for csv)
    """
    columns = []
    # Flockingパラメータ
    for keys in frc_dict.keys():
        if frc_dict[keys]:
            columns.extend(label_dict[keys])
    # 目的関数値
    if single_obj: # 単目的最適化の場合
        columns.append("objective")
    else: # 多目的最適化の場合
        for keys in obj_dict.keys():
            if obj_dict[keys]:
                columns.append(obj_label_dict[keys])
    return columns

def get_heatmap_matrix(x_num: int, y_num: int, x_limit: list, y_limit: list) -> list:
    heat_matrix = []
    x_ll = x_limit[0]
    x_ul = x_limit[1]
    y_ll = y_limit[0]
    y_ul = y_limit[1]
    for row in range(y_num):
        row_list = []
        for col in range(x_num):
            x_param = x_ll + (x_ul - x_ll)*col/(x_num-1)
            y_param = y_ll + (y_ul - y_ll)*row/(y_num-1) 
            row_list.append([x_param, int(y_param)])
        heat_matrix.append(row_list)
    return heat_matrix

def get_heatmap_matrix_loop(x_num: int, y_num: int, x_limit: list, y_limit: list, n_row: list) -> list:
    heat_matrix = []
    x_ll = x_limit[0]
    x_ul = x_limit[1]
    y_ll = y_limit[0]
    y_ul = y_limit[1]
    for row in range(y_num):
        row_list = []
        for col in range(x_num):
            x_param = x_ll + (x_ul - x_ll)*col/(x_num-1)
            y_param = y_ll + (y_ul - y_ll)*row/(y_num-1) 
            row_list.append([x_param, int(y_param), n_row])
        heat_matrix.append(row_list)
    return heat_matrix
