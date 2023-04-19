import numpy as np
import math

from settings import calc
from parameters import params

class Env:
    def __init__(self, driving_mode: str, n_follower: int, leader_pos: np.ndarray, leader_vel: np.ndarray):
        self.driving_mode = driving_mode # 走行モード{直進(straight), 車線変更(lane_change), 交差点(inter)}
        self.n_lane = params.N_LANE # [m] 車線の数
        self.w_lane = params.W_LANE # [m] 車線の幅
        self.n_follower = n_follower
        self.n_vehicle = params.N_LEADER + self.n_follower
        self.l_lane_list = params.INT_LIST # [m,m,m,m] 交差点周りの車線の長さのリスト[right, up, left, down]
        self.adj = 0.0 # [-] 
        self.leader_pos = leader_pos # [m]
        self.leader_vel = leader_vel # [m/s]
    
    def dec_drive_scene(self, pos):
        """直進・右折・左折などの走行シーンを選択
        """
        if self.driving_mode == "straight":
            scene = "STR"
            return scene
        
        elif self.driving_mode == "lane_change":
            scene = "LNC"
            return scene

        w_inter = self.n_lane*self.w_lane # width of intersection
        # if - w_inter < pos[0] < w_inter and - w_inter < pos[1] < w_inter: # Intersection
        if pos[0] < w_inter and - w_inter < pos[1]: # Intersection
            scene = "INT"
            # print(scene)
        # elif -w_inter < pos[0] < w_inter and pos[1] < - w_inter: # Down Lane
        elif pos[0] < w_inter and pos[1] < - w_inter: # Down Lane
            scene = "L_D"
            # print(scene)
        # elif w_inter < pos[0] and - w_inter < pos[1] < w_inter: # Right Lane
        elif w_inter < pos[0] and - w_inter < pos[1]: # Right Lane
            scene = "L_R"
            # print(scene)
        else:
            scene = "L_D"
            print("Driving Scene Error")
            print(f"pos:{pos}")
        # scene = "L_R"
        return scene
    
    def rotate_angle(self, o_xy, pos):
        """
        input:
            o_xy: array([x, y]) Rotate Origin
            pos: array([x, y]) Vehicle Position

        output:
            float: [rad] Rotate Angle
        """
        dis = np.linalg.norm(pos - o_xy) # np.array
        pos_dash = pos - o_xy
        rad_x = math.acos(pos_dash[0]/dis)
        rad_y = math.asin(pos_dash[1]/dis)
        dis -= self.n_lane*self.w_lane
        return dis, rad_x, rad_y

    def width_ratio(self, velocity):
        """
        input:
            velocity
                float: [m/s] leader velocity
            self.w_lane
                float: [m] w_lane # lane width
        output:
            float: [-] width ratio 
        """
        dis = calc.get_braking_distance(velocity=velocity, veh_dis=params.VEH_DIS)
        wid_ratio = dis/self.w_lane*np.sqrt(3/4)
        self.adj = wid_ratio
        # return width_ratio
    
    def calc_theta(self, pos):
        """
        input:
            pos
                vector[x,y]: [m] vehicle position
        """
        scene = self.dec_drive_scene(pos)
        if scene == "STR" or scene == "LNC":
            theta = 0.0
            return theta
        if scene == "L_R":
            theta = 0.0
        elif scene == "L_D":
            theta = np.pi/2
        elif scene == "INT":
            dis, _, theta = self.rotate_angle(o_xy=np.array([9,-9]), pos=pos)
        return theta

    def calc_lane(self, pos):
        """
        input:
            pos
                vector[x,y]: [m] vehicle position
        """
        scene = self.dec_drive_scene(pos)
        if self.driving_mode == "straight":
            theta = 0.0
            road_pos = pos[1]
        if scene == "L_R":
            theta = 0.0
            road_pos = pos[1]
        elif scene == "L_D":
            _theta = np.pi/2
            road_pos = abs(pos[0])
        elif scene == "INT":
            dis, _, theta = self.rotate_angle(o_xy=np.array([9,-9]), pos=pos)
            road_pos = dis
        return road_pos
