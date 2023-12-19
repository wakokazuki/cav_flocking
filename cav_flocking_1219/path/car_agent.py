import sys
import os
import numpy as np

# root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params

class TargetCourse:
    """ターゲット軌道
    """
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state: object) -> tuple[int, float]:
        """_summary_

        Args:
            state (object): _description_

        Returns:
            tuple[int, float]: _description_
        """
        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index == None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            dis_list = np.hypot(dx, dy) # 後軸(rear_axis)からの距離のリスト
            ind = np.argmin(dis_list)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind
        look_ahead_distance =  0.72 * state.velocity + 2.7  # [m] Pure-pursuitのLook ahead distance # 式は次の論文を参照(https://doi.org/10.1109/ICCAS.2014.6987822)
        look_ahead_distance = 4.0
        # search look ahead target point index
        while look_ahead_distance > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1
        return ind, look_ahead_distance

def adjust_control_input(accel: float, delta: float) -> tuple[float, float]:
    """制御入力を調整
    Args:
        accel (float): [m/s/s] 加速度
        delta (float): [rad] ステアリング角(車体に対する前輪タイヤの角度)
    Returns:
        tuple[float, float]: accel, delta
    """
    # 加速度の調整
    if accel > params.A_MAX:
        accel = params.A_MAX
    elif accel < params.A_MIN:
        accel = params.A_MIN
    # ステアリング角の調整
    if delta > params.DELTA_MAX:
        delta = params.DELTA_MAX
    elif delta < params.DELTA_MIN:
        delta = params.DELTA_MIN
    return accel, delta

def get_scalar_accel(desired_vel: float, current_vel: float) -> float:
    """加速度の大きさ(scalar)を取得
    Args:
        desired_vel (float): [m/s] 車両の希望速度
        current_vel (float): [m/s] 現在の車両の速度
    Returns:
        float: [m/s/s] 加速度の大きさ(scalar)
    """
    accel = min(desired_vel - current_vel, params.A_MAX)
    return accel

def pure_pursuit_steer_control(state: object, trajectory: object, pind: int) -> tuple[float, int]:
    """Pure-pursuitによる軌道追従制御
    Args:
        state (object): _description_
        trajectory (object): _description_
        pind (int): _description_
    Returns:
        tuple[float, int]: [rad] ステアリング角, [-] ターゲット軌道のindex
    """
    ind, Lf = trajectory.search_target_index(state)
    if pind >= ind:
        ind = pind
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1
    alpha = np.arctan2(ty - state.rear_y, tx - state.rear_x) - state.yaw # [rad] 車体方位とReference pointとのなす角度
    delta = np.arctan2(2.0 * params.WB * np.sin(alpha) / Lf, 1.0)
    return delta, ind

def get_straight_path(x: float, y: float, nt: int, velocity: float) -> tuple[np.ndarray, np.ndarray]:
    """直進の参照経路を取得
    Args:
        x (float): [m] 初期位置のx座標
        y (float): [m] 初期位置のy座標
        nt (int): [-] タイムステップ数
        velocity (float): [m/s] 速度
    Returns:
        tuple[np.ndarray, np.ndarray]: x座標のリスト, y座標のリスト
    """
    x_list = np.array([x + velocity*params.DT*i for i in range(nt+100)])
    y_list = np.array([y for _ in range(nt+100)])
    return x_list, y_list

def get_lanechange_path(init_pos: list, change_pos: list, path_x: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """車線変更の参照経路を取得 式は次の論文を参照(https://doi.org/10.1016/j.trc.2018.06.007)
    Args:
        x (float): [m] 車線変更をする距離
        path_x (float): [m]車線変更後の直線の距離
        y (float): [m] 車線の幅
        step (float): [m] 参照経路のポイント間の等差
    Returns:
        tuple[np.ndarray, np.ndarray]: x座標のリスト, y座標のリスト
    """
    [init_x, init_y] = init_pos
    [change_x, change_y] = change_pos
    a0 = 0
    a1 = 0
    a2 = 3*change_y / change_x**2
    a3 = -2*change_y / change_x**3
    x_list = np.arange(init_x, init_x+change_x, step)
    y_list = init_y + a2 * (x_list - init_x)**2 + a3 * (x_list - init_x)**3
    # 車線変更後の直線の参照経路
    count_str = int(path_x/step)
    x_add = [init_x+change_x + path_x/count_str*i for i in range(count_str)]
    y_add = [init_y + change_y for _ in range(count_str)]
    x_list = np.append(x_list, x_add)
    y_list = np.append(y_list, y_add)
    return x_list, y_list

def get_inter_path(max_vel: float, init_pos: list, final_pos: list) -> tuple[np.ndarray, np.ndarray]:
    def _inter_target(count: int, o_xy: list, radius: float, s_angle: float, e_angle: float):
        x_inter = []
        y_inter = []
        for i in range(count):
            rad = s_angle + (e_angle - s_angle)*i/count
            R = np.array([[np.cos(rad), -np.sin(rad)],
                            [np.sin(rad), np.cos(rad)]])
            _point = np.dot(R, np.array([radius, 0])) + np.array(o_xy)
            x_inter.append(_point[0])
            y_inter.append(_point[1])
        return x_inter, y_inter    
    dt = params.DT
    lane_width = params.N_LANE*params.W_LANE 
    enter_length = max(np.abs(init_pos)) - lane_width
    enter_int = int(enter_length/(max_vel*dt)) # 参照パスのカウント数
    x_enter = [init_pos[0] for _ in range(enter_int)]
    y_enter = [init_pos[1]+enter_length*i/enter_int for i in range(enter_int)]
    inter_int = round(np.pi*lane_width/ (max_vel*dt)) # 参照パスのカウント数
    print(inter_int)
    x_inter, y_inter = _inter_target(count=inter_int, o_xy=[lane_width, -lane_width], radius=lane_width+abs(init_pos[0]), s_angle=np.pi, e_angle=np.pi/2) 
    exit_length = max(np.abs(final_pos)) - lane_width + 20.0
    exit_int = int(exit_length/(max_vel*dt)) # 参照パスのカウント数
    x_exit = [lane_width+exit_length*i/exit_int for i in range(exit_int)] 
    y_exit = [final_pos[1] for _ in range(exit_int)]
    x_list = x_enter + x_inter + x_exit
    y_list = y_enter + y_inter + y_exit
    return x_list, y_list


def intersection(inter_list: list):
    direc_num = params.INT_DIREC_NUM # 交差点の方向の数(固定)
    lane_num = params.N_LANE
    lane_width = params.W_LANE
    total_width = lane_num*lane_width
    len_list = inter_list
    # 車線の中央線
    _center = []
    for i in range(direc_num):
        R = np.array([[np.cos(i*np.pi/2), -np.sin(i*np.pi/2)],
                      [np.sin(i*np.pi/2), np.cos(i*np.pi/2)]])
        _start = np.array([total_width, 0])
        _center.append(np.dot(R, _start.T))
        _end = np.array([total_width+len_list[i], 0])
        _center.append(np.dot(R, _end.T))
    _center = np.array(_center).T
    # 車線
    _lane = []
    for i in range(direc_num):
        R = np.array([[np.cos(i*np.pi/2), -np.sin(i*np.pi/2)],
                      [np.sin(i*np.pi/2), np.cos(i*np.pi/2)]])
        _start = np.array([total_width, 0])
        _end = np.array([total_width+len_list[i], 0])
        for la in range(lane_num-1):
            _upper = _start + np.array([0, lane_width*(la+1)])
            _lane.append(np.dot(R, _upper.T))
            _upper = _end + np.array([0, lane_width*(la+1)])
            _lane.append(np.dot(R, _upper.T))
            _lower = _start - np.array([0, lane_width*(la+1)])
            _lane.append(np.dot(R, _lower.T))
            _lower = _end - np.array([0, lane_width*(la+1)])
            _lane.append(np.dot(R, _lower.T))
    _lane = np.array(_lane).T
    # 道路端
    _road_side = []
    for i in range(direc_num):
        R = np.array([[np.cos(i*np.pi/2), -np.sin(i*np.pi/2)],
                      [np.sin(i*np.pi/2), np.cos(i*np.pi/2)]])
        _start = np.array([total_width, 0])
        _end = np.array([total_width+len_list[i], 0])
        _upper = _start + np.array([0, total_width])
        _road_side.append(np.dot(R, _upper))
        _lower = _end + np.array([0, total_width])
        _road_side.append(np.dot(R, _lower))
        _upper = _start - np.array([0, total_width])
        _road_side.append(np.dot(R, _upper))
        _lower = _end - np.array([0, total_width])
        _road_side.append(np.dot(R, _lower))
    _road_side = np.array(_road_side).T
    return _center, _lane, _road_side