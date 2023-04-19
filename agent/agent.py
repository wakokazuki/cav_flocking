import numpy as np
import copy

from algorithm import boids, boids_vl
import parameters.params as params
from settings import calc
from car_following_model import car_following

class Agent:
    """車両エージェントを定義
    """
    def __init__(self, state: str, set_params: list, max_vel: float, init_pos: list) -> None:
        # エージェントの設定
        self.state = state # 車両の状態("L":リーダー, "V":仮想リーダー, "F":Flocking車両, "C":追従車両(人間の運転する車両))
        self.max_vel = max_vel # [m/s] 車両の最大速度
        self.des_dis = calc.get_braking_distance(velocity=max_vel, veh_dis=params.VEH_DIS) # [m] 希望車間距離
        self.cls_dis = np.sqrt(2)*self.des_dis # クラスターの距離(目的関数の評価用)
        # エージェントの状態
        self.pos = np.array(init_pos)  # [m] 現在の車両の位置(x,y) 
        self.pre_pos = copy.deepcopy(self.pos)  # [m] 1ステップ前の車両の位置(x,y)
        self.velocity = max_vel  # [m/s] 現在の車両の速度
        self.pre_velocity = copy.deepcopy(self.velocity) # [m/s] 直前の車両の速度
        self.accel = 0.0 # [m/s/s] 車両の加速度
        self.yaw = 0.0 # [rad] 車両のヨー角
        self.slip = 0.0 # [rad] 車両の横滑り角
        self.delta = 0.0 # [rad] 車両のステアリング角
        self.theta = self.yaw + self.slip # [rad] 車両の速度ベクトル角
        # 車両の状態ごとの設定
        if self.state == "L" or self.state == "V": # リーダーの場合
            self.leader_pos = None
            self.leader_vel = None
        if self.state == "F": # Flocking車両の場合
            self.set_params = set_params # Flockingアルゴリズムのパラメータセット
        elif self.state == "C": # 追従車両の場合
            self.rear_x = self.pos[0] - ((params.WB / 2) * np.cos(self.yaw))
            self.rear_y = self.pos[1] - ((params.WB / 2) * np.sin(self.yaw))
            self.next_lane = None # 車線変更先の車線 (上側: +1, 下側: -1)
            self.lane_change_start_pos = None # 車線変更開始地点の位置座標
            self.driving_mode = None # 車両の走行モード ("lane_change" or "following")
            self.init_flag = True 

    def _calc_next(self, num: int, agents: object, env: object, t: int) -> None:
        """車両の加速度を計算
        Args:
            num (int): 車両の番号
            agents (object): エージェントクラス
            env (object): 環境クラス
            t (int): 現在のタイムステップ
        """
        if self.state == "F": # Flocking車両
            self._state_F(num, agents, env)  
        elif self.state == "L": # リーダー
            self._state_L(num, agents, env, t)
        elif self.state == "V": # 仮想リーダー
            self._state_V(num, agents, env, t)
        elif self.state == "C": # 追従車両
            self._state_C(num, agents)
        else:  # 合致するカテゴリがない
            print("ERROR. No category")
    
    def calc_distance(self, point_x: float, point_y: float) -> float:
        """車両の中心点から後軸までの距離を計算
        Args:
            point_x (float): 中心点のx座標
            point_y (float): 中心点のy座標
        Returns:
            float: 距離
        """
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return np.hypot(dx, dy)

    def _state_F(self, num: int, agents: object, env: object) -> None: # Flocking車両
        """Flocking車両の速度・位置の更新
        Args:
            num (int): 車両の番号
            agents (object): エージェントクラス
            env (object): 環境クラス
        """
        # Boidsアルゴリズムを選択
        if params.VIRTUAL_LEADER: # 仮想リーダーの場合
            BoidsAgents = boids_vl.Boids(agents, env)
        else: # リーダーの場合
            BoidsAgents = boids.Boids(agents, env)
        # 加速度を計算
        a_vector = BoidsAgents._calc_a(num)
        accel, delta = self._adjust_control_input(a_vector=a_vector)        
        # 位置座標を更新
        self._update_kbm_center(accel=accel, delta=delta)

    def _state_L(self, num, agents, env, t) -> None: # リーダー
        """リーダーの速度・位置の更新
        Args:
            num (int): 車両の番号
            agents (object): エージェントクラス
            env (object): 環境クラス
            t (int): 現在のタイムステップ
        """
        # 車両幅方向の調整
        env.width_ratio(agents[num].velocity)
        # 位置座標を更新
        self.pos = self.leader_pos[t]
        self.velocity = np.linalg.norm(self.leader_vel[t])
    
    def _state_V(self, num, agents, env, t) -> None: # 仮想リーダー
        """仮想リーダーの速度・位置の更新
        Args:
            num (int): 車両の番号
            agents (object): エージェントクラス
            env (object): 環境クラス
            t (int): 現在のタイムステップ
        """
        # 車両幅方向の調整
        env.width_ratio(agents[num].velocity)
        # 位置座標を更新
        self.pos = self.leader_pos[t]
        self.velocity = np.linalg.norm(self.leader_vel[t])
    
    def _state_C(self, num: int, agents: object) -> None: # 追従車両
        """追従車両の速度・位置の更新
        Args:
            num (int): 車両の番号
            agents (object): エージェントクラス
        """
        # 車線前方の物体(車両 or 障害物)の取得
        lead_name = car_following.get_lead_name(agents=agents, vehicle_index=num)
        # 加速度を計算
        accel, delta = car_following.get_control_input(lead_name=lead_name, agents=agents, vehicle_index=num)
        # 位置座標を更新
        self._update_kbm_rear(accel=accel, delta=delta)
        
    def _update_state(self) -> None:
        """位置・速度の更新
        """
        self.pre_pos = copy.deepcopy(self.pos)
        self.pre_velocity = copy.deepcopy(self.velocity)

    def _adjust_control_input(self, a_vector: np.ndarray) -> tuple[float, float]:
        """速度・加速度・ステアリング角速度の調整
        Args:
            a_vector (np.ndarray): 調整前の加速度ベクトル
        Returns:
            tuple[float, float]: [車両の加速度, 車両の操舵角]
        """
        def _accel(accel): # 加速度の調整
            if accel <= params.A_MIN:
                accel = params.A_MIN
            elif accel >= params.A_MAX:
                accel = params.A_MAX
            return accel
        def _velocity(vel): # 速度の調整
            if vel <= 0.0:
                vel = 0.0
            elif vel >= self.max_vel: 
                vel = self.max_vel
            return vel
        def _steering_vel(steer): # ステアリング角速度の調整
            if steer <= params.PHI_MIN:
                steer = params.PHI_MIN
            elif steer >= params.PHI_MAX:
                steer = params.PHI_MAX
            return steer
        dt = params.DT
        v_vector = self._velocity_vector()
        des_v_vector = v_vector + a_vector*dt            
        # 速度ベクトルの角度の調整
        des_v_angle = np.arctan2(des_v_vector[1], des_v_vector[0]) 
        slip_velocity, slip_angle = self._adjust_slip_velocity_angle(slip_velocity=np.linalg.norm(des_v_vector), velocity_angle=des_v_angle)
        # 速度の大きさ調整
        slip_velocity = _velocity(vel=slip_velocity)
        # 加速度の調整
        slip_accel = (slip_velocity - self.velocity)/dt
        slip_accel = _accel(accel=slip_accel)
        # 躍度の調整
        slip_jerk = (slip_accel - self.accel)/dt
        if slip_jerk <= params.JERK_MIN:
            slip_jerk = params.JERK_MIN
        elif slip_jerk >= params.JERK_MAX:
            slip_jerk = params.JERK_MAX
        # 躍度から加速度への変換
        accel = self.accel + slip_jerk*dt
        accel = _accel(accel=accel)
        velocity = self.velocity + accel*dt
        velocity = _velocity(vel=velocity)
        accel = (velocity - self.velocity)/dt
        # 横滑り角からステアリング角への変換
        delta = slip_angle
        # ステアリング角の調整
        steer_vel = (delta - self.delta)/dt
        steer_vel = _steering_vel(steer=steer_vel) 
        delta = self.delta + steer_vel*dt
        return accel, delta
    
    def _adjust_slip_velocity_angle(self, slip_velocity: float, velocity_angle: float) -> tuple[float, float]:
        """車両速度[m/s]と横滑り角[rad]をステアリング角の制限により調整
        Args:
            slip_velocity (float): [m/s] 調整前の車両速度
            velocity_angle (float): [rad] 調整前の車両の速度ベクトル
        Returns:
            tuple[float, float]: 車両速度, 横滑り角
        """
        slip_angle = velocity_angle - self.yaw
        # 横滑り角の調整
        if slip_angle >= np.pi:
            slip_angle -= np.pi
        elif slip_angle <= -np.pi:
            slip_angle += np.pi
        # ステアリング角への変換
        if slip_angle >= params.DELTA_MAX: # 横滑り角が正で制限以上の場合
            if slip_angle <= np.pi - params.DELTA_MAX: # 第一象限
                slip_velocity = slip_velocity*np.cos(slip_angle-params.DELTA_MAX)
                slip_angle = params.DELTA_MAX
            elif slip_angle <= np.pi - params.DELTA_MAX: # 第二象限
                slip_velocity = -slip_velocity*np.cos(np.pi - params.DELTA_MAX - slip_angle)
                slip_angle = params.DELTA_MAX
            else: # 第二象限の制限以内
                slip_velocity = -slip_velocity
                slip_angle = np.pi - slip_angle
        elif slip_angle <= params.DELTA_MIN: # 横滑り角が負で制限以上の場合
            # print(f"Angle Min, slip:{slip_angle}")
            if slip_angle >= -np.pi - params.DELTA_MIN: # 第四象限
                slip_velocity = slip_velocity*np.cos(slip_angle-params.DELTA_MIN)
                slip_angle = params.DELTA_MIN
            elif slip_angle >= -np.pi - params.DELTA_MIN: # 第三象限
                slip_velocity = -slip_velocity*np.cos(-np.pi - params.DELTA_MIN - slip_angle)
                slip_angle = params.DELTA_MIN
            else: # # 第三象限の制限以内
                slip_velocity = -slip_velocity
                slip_angle = -np.pi - slip_angle
        return slip_velocity, slip_angle

    def _velocity_vector(self) -> np.ndarray:
        """速度ベクトルを計算
        Returns:
            np.ndarray: [m/s] 速度ベクトル [x,y]
        """
        vector = np.array([self.pre_velocity*np.cos(self.theta), self.pre_velocity*np.sin(self.theta)])
        return vector
    
    def _update_kbm_rear(self, accel: float, delta: float) -> None:
        """Kinematic Bycycle Model(KBM)による後軸(rear)の位置・速度の更新
        Args:
            accel (float): [m/s/s] 車両の加速度
            delta (float): [rad] 車両の操舵角
        """
        dt = params.DT
        self.pos[0] += self.velocity * np.cos(self.yaw) * dt # x座標
        self.pos[1] += self.velocity * np.sin(self.yaw) * dt # y座標
        self.yaw += self.velocity / params.WB * np.tan(delta) * dt
        self.velocity += accel * dt
        self.rear_x = self.pos[0] - ((params.WB / 2) * np.cos(self.yaw))
        self.rear_y = self.pos[1] - ((params.WB / 2) * np.sin(self.yaw))
        self.accel = accel
        self.delta = delta
        self.theta = self.yaw + self.slip # [rad] 速度ベクトルの角度
    
    def _update_kbm_center(self, accel: float, delta: float) -> None:
        """Kinematic Bycycle Modelによる車両中心(center)の位置・速度の更新
        Args:
            accel (float): [m/s/s] 車両の加速度
            delta (float): [rad] 車両の操舵角
        """
        dt = params.DT
        self.pos[0] += self.velocity*np.cos(delta + self.yaw) * dt # x座標
        self.pos[1] += self.velocity*np.sin(delta + self.yaw) * dt # y座標
        self.slip = np.arctan(np.tan(delta)/2)
        self.yaw += self.velocity * np.tan(delta) * np.cos(self.slip) / params.WB * dt
        self.velocity += accel * dt
        self.rear_x = self.pos[0] - ((params.WB / 2) * np.cos(self.yaw))
        self.rear_y = self.pos[1] - ((params.WB / 2) * np.sin(self.yaw))
        self.accel = accel
        self.delta = delta
        self.theta = self.yaw + self.slip # [rad] 速度ベクトルの角度
    
class StateList:
    """車両エージェントの状態リスト
    """
    def __init__(self, agent_num: int) -> None:
        self.pos = [[] for _ in range(agent_num)] # 位置
        self.velocity = [[] for _ in range(agent_num)] # 速度
        self.accel = [[] for _ in range(agent_num)] # 加速度
        self.yaw = [[] for _ in range(agent_num)] # ヨー角
        self.delta = [[] for _ in range(agent_num)] # ステアリング角
        self.slip = [[] for _ in range(agent_num)] # 横滑り角
        self.start_time = [None for _ in range(agent_num)] # 開始時間
        self.end_time = [None for _ in range(agent_num)] # 終了時間

    def _append(self, num: int, vehicle: object) -> None:
        """現在の状態をリストに追加
        Args:
            num (int): (リーダーを除いた)車両の番号
            vehicle (object): 車両クラス
        """
        self.pos[num].append([round(vehicle.pos[0], 3), round(vehicle.pos[1],3)]) # list
        self.yaw[num].append(round(vehicle.yaw, 3)) # float
        self.velocity[num].append(round(vehicle.velocity, 3)) # float
        self.accel[num].append(round(vehicle.accel, 3))
        self.delta[num].append(round(vehicle.delta, 3))
        self.slip[num].append(round(vehicle.slip, 3))
    
    def _check_pos(self, num: int, vehicle: object, t:int): # 空間平均速度計算用
        """空間平均速度計算のために, 車両の位置が領域内にあるか判定
        Args:
            num (int): 車両の番号
            agents (object): エージェントクラス
            env (object): 環境クラス
            t (int): 現在のタイムステップ
        """
        if self.start_time[num] == None: # 開始時間を記録
            if vehicle.pos[0] >= params.SPACE_LIST[0]:
                self.start_time[num] = t
        elif self.end_time[num] == None: # 終了時間を計算
            if vehicle.pos[0] >= params.SPACE_LIST[1]:
                self.end_time[num] = t
    
    def _calc_space_mean_speed(self, agent_num: int) -> float:
        """空間平均速度を計算
        Args:
            agent_num (int): 車両の番号
        Returns:
            float: [m/s] 空間平均速度
        """
        sum_space_length = agent_num * (max(params.SPACE_LIST) - min(params.SPACE_LIST)) # 全車両の総走行距離
        sum_vehicle_time = 0.0 # 全車両の総走行時間
        for i in range(agent_num):
             vehicle_time =  (self.end_time[i] - self.start_time[i])*params.DT # 車両の総走行時間
             sum_vehicle_time += vehicle_time
        space_mean_speed = sum_space_length / sum_vehicle_time # 空間平均速度
        return space_mean_speed
    
    def _judge_pos(self) -> bool:
        """全ての車両が指定空間を通過したかの判定
        Returns:
            bool: 空間を通過したかの判定
        """
        for x in self.end_time:
            if x == None:
                return False
        return True
