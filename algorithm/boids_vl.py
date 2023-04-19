import numpy as np
import copy

from parameters import params
from settings import set, calc

class Boids:
    """define flocking rules
    """
    def __init__(self, agents, env):
        self.agents = agents
        self.env = env

    def _calc_a(self, num: int) -> np.ndarray: # 加速度ベクトル(a_des)を決定
        """calc accel by flocking algorithm
        Args:
            num (int): 車両の番号
        Returns:
            float: acceleration of agent
        """
        # init
        [separation_vec, alignment_vec, attraction_vec, lane_vec, navi_vec, obstacle_vec] = [np.zeros(2) for _ in range(len(params.FRC_LIST))]
        #  Flockingルールの加速度ベクトルを計算
        if params.SEP_FRC:
            separation_vec = self._separation(num) # 分離(separation)
        if params.ALI_FRC:
            alignment_vec = self._alignment(num) # 整列(alignment)
        if params.ATT_FRC:
            attraction_vec = self._attraction(num) # 結合(cohesion)
        if params.LAN_FRC:
            lane_vec = self._lane(num) # 車線幅(lane)
        if params.NAV_FRC:
            navi_vec = self._navigation(num) # 誘導(navigation)
        if params.OBS_FRC:
            if params.IS_OBS_POTENTIAL:
                obstacle_vec = self._obstacle_velocity_potential(num) # 障害物(obstacle)-速度ポテンシャル
            else:
                obstacle_vec = self._obstacle_repulsion(num) # 障害物(obstacle)-反発力 
        # 加速度ベクトルを計算
        a_vector = separation_vec + alignment_vec + attraction_vec + lane_vec + navi_vec + obstacle_vec
        return a_vector
    
    def _velocity_func(self, radius: float, accel: float, linear_gain: float) -> float:
        """a smooth decay function by Vasarhelyi et al. 2018
        """
        rp = radius*linear_gain
        a_p = accel / linear_gain
        if radius <= 0:
            _v_vfunc = 0
        elif 0 < rp and rp < a_p:
            _v_vfunc = rp
        else:
            _v_vfunc  = np.sqrt(2*accel*radius - accel**2 / (linear_gain**2))
        return _v_vfunc
    
    def _separation(self, num: int) -> np.ndarray: # 分離(separation)
        vehicle = self.agents[num] # 車両
        separation_vec = np.zeros(2) # vector[x,y] 
        total = 0
        [r0_rep, p_rep] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="SEP")
        theta = self.env.calc_theta(pos=vehicle.pre_pos)
        rotate_matrix, inverse_matrix = calc.get_rotate_matrix(ang=theta)
        for i in range(self.env.n_vehicle):
            if i != num and self.agents[i].state != "V":
                y_axis = 1 # y-axis order
                pos_diff = vehicle.pre_pos - self.agents[i].pre_pos
                distance, rotate_pos_diff = calc.get_adjusted_difference(y_axis=y_axis, adj=self.env.adj, inverse_matrix=inverse_matrix, difference=pos_diff)
                if distance < r0_rep:
                    total += 1
                    separation_vec += p_rep * ( r0_rep - distance ) * rotate_pos_diff / distance
        if total > 0:
            separation_vec /= total 
        separation_vec[y_axis] /= self.env.adj
        separation_vec = np.dot(rotate_matrix, separation_vec.T)
        return separation_vec

    def _alignment(self, num: int) -> np.ndarray: # 整列(alignment)
        vehicle = self.agents[num] # 車両
        alignment_vec = np.zeros(2) # vector[x,y]
        [r0_frict, p_frict, v_frict] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="ALI")
        a_frict, p_frict = [3.0, 3.0]
        theta = self.env.calc_theta(vehicle.pre_pos)
        rotate_matrix, inverse_matrix = calc.get_rotate_matrix(ang=theta)
        for i in range(self.env.n_vehicle):
            if i != num and self.agents[i].state == "F":
                y_axis = 1 # y-axis order
                pos_diff = vehicle.pre_pos - self.agents[i].pre_pos
                distance, _ = calc.get_adjusted_difference(y_axis=y_axis, adj=self.env.adj, inverse_matrix=inverse_matrix, difference=pos_diff)
                vel_diff = vehicle._velocity_vector() - self.agents[i]._velocity_vector() # velocity difference
                v_ij, rotate_vel_diff = calc.get_adjusted_difference(y_axis=y_axis, adj=self.env.adj, inverse_matrix=inverse_matrix, difference=vel_diff)
                # D() in Vasarhelyi et al. 2018 
                d_func = self._velocity_func(distance - r0_frict , a_frict, p_frict)
                v_frict_max = max(v_frict, d_func)
                if distance < params.CAV_DIS and v_ij > v_frict_max:
                    alignment_vec += p_frict * (v_ij - v_frict_max) * (- rotate_vel_diff ) / v_ij
        alignment_vec[y_axis] /= self.env.adj
        alignment_vec = np.dot(rotate_matrix, alignment_vec.T)
        return alignment_vec

    def _attraction(self, num: int) -> np.ndarray: # 結合(cohesion)
        vehicle = self.agents[num] # 車両
        attraction_vec = np.zeros(2) # vector[x,y] 
        total = 0
        [p_att] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="ATT")
        theta = self.env.calc_theta(vehicle.pre_pos)
        rotate_matrix, inverse_matrix = calc.get_rotate_matrix(ang=theta)
        for i in range(self.env.n_vehicle):
            if i != num and self.agents[i].state == "F":
                y_axis = 1 # y-axis order
                pos_diff = vehicle.pre_pos - self.agents[i].pre_pos
                distance, rotate_pos_diff = calc.get_adjusted_difference(y_axis=y_axis, adj=self.env.adj, inverse_matrix=inverse_matrix, difference=pos_diff)
                if distance > vehicle.des_dis and distance <= params.CAV_DIS:
                    total += 1
                    attraction_vec += p_att * (vehicle.des_dis - distance) * rotate_pos_diff / distance
        if total > 0:
            attraction_vec /= total 
        attraction_vec[y_axis] /= self.env.adj
        attraction_vec = np.dot(rotate_matrix, attraction_vec.T)
        return attraction_vec
    
    def _lane(self, num: int) -> np.ndarray: # 車線幅(lane)
        vehicle = self.agents[num] # 車両
        lane_vec = np.zeros(2) # vector[x,y]
        y_axis = 1
        [p_edge, p_lane] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="LAN")
        theta = self.env.calc_theta(vehicle.pre_pos)
        rotate_matrix, _ = calc.get_rotate_matrix(ang=theta)
        # 道路端から受ける力
        road_pos = self.env.calc_lane(vehicle.pre_pos)
        half_road = params.SIZE_Y/2 
        to_center = half_road - road_pos
        dis_from_wall = half_road - abs(to_center)
        if dis_from_wall <= params.ROAD_DIS:
           lane_vec[y_axis] += p_edge * (params.ROAD_DIS - dis_from_wall) * to_center/abs(to_center) 
        # 車線から受ける力
        lane_vec[y_axis] += p_lane*np.pi/params.W_LANE*(np.sin(2*np.pi*road_pos/params.W_LANE))
        lane_vec = np.dot(rotate_matrix, lane_vec.T)
        return lane_vec

    def _navigation(self, num: int) -> np.ndarray: # 誘導(navigation)
        """Navigation Feedback by Olfati-Saber
        """
        vehicle = self.agents[num] # 車両
        navigation_vec = np.zeros(2) # vector[x, y]
        [p_pos, p_vel] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="NAV")
        theta = self.env.calc_theta(vehicle.pre_pos)
        rotate_matrix, inverse_matrix = calc.get_rotate_matrix(ang=theta)
        leader_pos = self.agents[params.ORD_LEADER].pre_pos
        # leader_vel = self.agents[params.ORD_LEADER].pre_velocity
        leader_vel = self.agents[params.ORD_LEADER]._velocity_vector()
        y_axis = 1 # y-axis order
        pos_diff = vehicle.pre_pos - leader_pos
        _, rotate_pos_diff = calc.get_adjusted_difference(y_axis=y_axis, adj=self.env.adj, inverse_matrix=inverse_matrix, difference=pos_diff)
        # vel_diff = vehicle.pre_velocity-leader_vel
        vel_diff = vehicle._velocity_vector() - leader_vel
        _, rotate_vel_diff = calc.get_adjusted_difference(y_axis=y_axis, adj=self.env.adj, inverse_matrix=inverse_matrix, difference=vel_diff)
        navigation_vec += -p_pos*self.sigma_func(rotate_pos_diff)
        navigation_vec += -p_vel*(rotate_vel_diff)
        navigation_vec[y_axis] /= self.env.adj
        navigation_vec = np.dot(rotate_matrix, navigation_vec.T)
        return navigation_vec

    def _obstacle_repulsion(self,num: int) -> np.ndarray:
        """障害物はReplusionと同様に定義する
        """
        vehicle = self.agents[num] # 車両
        obstacle_vec = np.zeros(2) # vector[x,y]
        total = 0 # init
        adj = self.env.adj
        y_axis = 1
        [p_obs] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="OBS")
        if params.IS_OBSTACLE:
            for keys in params.OBS_DICT.keys():
                if params.OBS_DICT[keys]:
                    if calc.judge_obstacle_zone(agent_pos=vehicle.pre_pos, obs_zone=params.OBS_ZONE_DICT[keys]):
                        pos_diff = vehicle.pre_pos - params.OBS_POS_DICT[keys]
                        pos_diff[y_axis] *= adj 
                        distance = np.linalg.norm(pos_diff)
                        if distance < params.OBS_NORTICE_DIS:
                            total += 1
                            obstacle_vec += p_obs *( params.OBS_NORTICE_DIS - distance ) * (pos_diff) / distance
            if total > 0:
                obstacle_vec /= total
            obstacle_vec[y_axis] /= self.env.adj
        return obstacle_vec

    def _obstacle_velocity_potential(self, num: int) -> np.ndarray:
        vehicle = self.agents[num] # 車両
        obstacle_vec = np.zeros(2) # 加速度ベクトル[x,y]
        [p_obs] = set.get_flock_parameter(set_params=vehicle.set_params, frc_key="OBS")
        if params.IS_OBSTACLE:
            vel_vec = np.zeros(2) # 障害物ごとの加速度ベクトル
            is_zone = False
            for keys in params.OBS_DICT.keys():
                if params.OBS_DICT[keys]:
                    if calc.judge_obstacle_zone(agent_pos=vehicle.pre_pos, obs_zone=params.OBS_ZONE_DICT[keys]):
                        is_zone = True
                        obs_pos = copy.deepcopy(params.OBS_POS_DICT[keys])
                        obs_rad = copy.deepcopy(params.RAD_OBS)
                        obs_pos, obs_rad = calc.adjust_obstacle_parameter(pos=obs_pos, rad=obs_rad)
                        pos_diff = vehicle.pre_pos - obs_pos
                        pos_diff[0] /= params.LANE_RATE
                        leader_vel = self.agents[params.ORD_LEADER]._velocity_vector()[0]
                        vel_u = leader_vel*(1-obs_rad**2/(np.linalg.norm(pos_diff)**2)*np.cos(2*np.arctan2(pos_diff[1], pos_diff[0])))
                        vel_v = -leader_vel*obs_rad**2/(np.linalg.norm(pos_diff)**2)*np.sin(2*np.arctan2(pos_diff[1], pos_diff[0]))
                        vel_vec += np.array([vel_u, vel_v])
            if is_zone:
                vel_vec = max(np.linalg.norm(vel_vec), vehicle.max_vel)*vel_vec/np.linalg.norm(vel_vec)
                v_vector = vehicle._velocity_vector()
                obstacle_vec = vel_vec - v_vector
                obstacle_vec *= p_obs
        return obstacle_vec
    
    def sigma_func(self, z): # z:vector [x,y]
        """For navigation feedback(vector)
        """
        sigma_1 = z/np.sqrt(1+np.linalg.norm(z)**2)
        return sigma_1