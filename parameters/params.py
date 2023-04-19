"""プログラムの設定パラメータを定義
"""

import numpy as np
import os
import sys
from multiprocessing import cpu_count

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from settings import set, calc

# 道路環境の設定
N_LANE = 3 # [-] 車線数
# N_LANE = 2 # [-] 車線数
W_LANE = 3.0 # [m] 車線の幅
SIZE_X = 300  # [m] 空間のサイズ(X軸)
SIZE_Y = N_LANE*W_LANE  # [m] 空間のサイズ(Y軸)
NT = 250 # [-] 時間ステップの数 # 250, 375, 750
DT = 0.1  #[s] 時間ステップ
T_LIST = [0, 50, 100, 150, 200, 250, 300, 350, 400] # 

POS_LIST = [[-24.53, 5.08], [-17.05, 3.86], [-12.58, 7.21], [-26.91, 7.63], [-16.15, 2.72], [-7.17, 1.92]] # 修論で使用した位置リスト

# 空間平均速度の計算
SPACE_LIST = [0.0, SIZE_X] # [m,m] 空間平均速度(Space Mean Speed)計算時の空間(Min, Max)


# 交差点の設定
INT_LIST = [20, 20, 20, 20] # 交差点周りの車線の長さのリスト[right, up, left, down]
INT_DIREC_NUM = 4 # 交差点の方向の数(固定)

# 車両の設定
N_LEADER = 1 # [-] リーダーの数
N_FOLLOWER = 9 # [-] Flocking車両の数
N_HUMAN = 4 # [-] 人間のドライバー
N_VEHICLE = N_LEADER + N_FOLLOWER  # 全車両の数
N_INTER = 6 # 交差点の車両の数
COLORS = ["tab:red", "tab:brown", "tab:purple", "tab:blue", "tab:green", "tab:olive", "tab:cyan", "tab:orange", "tab:pink", "tab:gray"] # 車両の色設定
L_COLORS = COLORS[0]
F_COLORS = COLORS[N_LEADER:N_VEHICLE]
ORD_LEADER = 0 # [-] リスト内のリーダーの順番
L_POS = [0.0, SIZE_Y/2] # [m,m] リーダーの初期位置
I_POS = [-4.5, -29.0] # [m,m] リーダーの初期位置(交差点)
F_POS = [29.0, 4.5] # [m,m] リーダーの最終位置(交差点)

# Flocking車両の設定
DELTA_MAX = np.pi/8 # [rad] 最大ステアリング角
DELTA_MIN = -np.pi/8 # [rad] 最小ステアリング角
PHI_MAX = DELTA_MAX # [rad/s] 最大ステアリング角速度
PHI_MIN = DELTA_MIN # [rad/s] 最小ステアリング角速度
A_MAX = 2.0 # [m/s^2] 最大加速度
A_MIN = -5.0 # [m/s^2] 最小加速度
V_MAX = 12.0 # [m/s] 最大速度 # 4 [m/s] = 14.4 [km/h], 8 [m/s] = 28.8 [km/h], 12 [m/s] = 43.2 [km/h]
V_MIN = 0.0 # [m/s] 最小速度
JERK_MAX = A_MAX # [m/s^3] 最大躍度
JERK_MIN = A_MIN # [m/s^3] 最小躍度



# 車両の大きさ(トヨタのプリウス)
CL = 4.5 # [m] 全長
CW = 1.8 # [m] 全幅
WB = 2.7 # [m] ホイールベース
DG = np.hypot(CL, CW) # [m] 車両の対角線の長さ
VEH_DIS = CL*2 # [m] 停止時の車間距離

# 道路の設定
ROAD_DIS = W_LANE/2 # 道端から受ける力(lane_vec)の影響範囲

# Flockingの評価の設定
COLL_DIS = CL # [m] 衝突距離
WALL_DIS = CW/2 # [m] 壁との衝突距離
OBS_DIS = CL # [m] 障害物・壁との衝突距離
CAV_DIS = 30 # [m] 自動運転車のセンシング半径 
OBS_NORTICE_DIS = 50 # [m] 障害物の感知半径
DES_DIS = calc.get_braking_distance(velocity=V_MAX, veh_dis=VEH_DIS) # 希望車間距離(制動距離+停止時の車間距離)
CLS_DIS = np.sqrt(2)*DES_DIS # クラスターの半径

# 障害物の設定
[IS_OBS1, IS_OBS2, IS_OBS3] = [True, True, True]
OBS_BOOL_LIST = [IS_OBS1, IS_OBS2, IS_OBS3] 
OBS_NUM = OBS_BOOL_LIST.count(True) # [-] 障害物の数
OBS_DICT = {"OBS1":IS_OBS1, "OBS2":IS_OBS2, "OBS3":IS_OBS3}
if N_LANE == 3:
    OBS_POS_DICT = {"OBS1":np.array([100, 1.5]), "OBS2":np.array([150, 7.5]), "OBS3":np.array([200, 4.5])} # [x,y]
elif N_LANE == 2:
    OBS_POS_DICT = {"OBS1":np.array([50, 1.5]), "OBS2":np.array([150, 4.5]), "OBS3":np.array([250, 1.5])} # [x,y]
OBS_ZONE_DICT = {"OBS1":calc.get_obstacle_zone(obs_pos=OBS_POS_DICT["OBS1"]),
                 "OBS2":calc.get_obstacle_zone(obs_pos=OBS_POS_DICT["OBS2"]),
                 "OBS3":calc.get_obstacle_zone(obs_pos=OBS_POS_DICT["OBS3"])
                 }
RAD_OBS = W_LANE/2 # [m] 障害物の半径
LANE_RATE = CAV_DIS # [-] 車線方向に障害物の半径を拡張する割合

"""Flocking Parameters
"""
# Flockingパラメータのラベル
sep_label = ["r0_rep", "p_rep"]
ali_label = ["r0_frict", "p_frict", "v_frict"]
att_label = ["p_att"]
lan_label = ["p_edge", "p_lane"]
navi_label = ["p_pos", "p_vel"]
obs_label = ["p_obs"]
LABEL_DICT = {"SEP":sep_label, "ALI":ali_label, "ATT":att_label, "LAN":lan_label, "NAV":navi_label, "OBS":obs_label}

# Flockingパラメータの下限(lower limit, ll)と上限(upper limit, ul)
sep_ll = [DES_DIS, 0.1]
sep_ul = [CLS_DIS, 1.5]
ali_ll = [DES_DIS/2, 0.01, 0.20]
ali_ul = [DES_DIS  , 0.20, 0.50]
att_ll = [0.01]
att_ul = [0.50]
lan_ll = [2.0, 0.01]
lan_ul = [5.0, 2.0]
nav_ll = [0.01, 0.1]
nav_ul = [0.50, 1.0]
obs_ll = [0.1]
obs_ul = [1.0]

L_LIM_DICT = {"SEP":sep_ll, "ALI":ali_ll, "ATT":att_ll, "LAN":lan_ll, "NAV":nav_ll, "OBS":obs_ll}
U_LIM_DICT = {"SEP":sep_ul, "ALI":ali_ul, "ATT":att_ul, "LAN":lan_ul, "NAV":nav_ul, "OBS":obs_ul}

# Flockingのルール一覧(Force, FRC)
[SEP_FRC, ALI_FRC, ATT_FRC, LAN_FRC, NAV_FRC, OBS_FRC] = [True, True, True, True, True, True]
FRC_LIST = [SEP_FRC, ALI_FRC, ATT_FRC, LAN_FRC, NAV_FRC, OBS_FRC]
FRC_BIN_DICT = {"SEP":int(SEP_FRC), "ALI":int(ALI_FRC), "ATT":int(ATT_FRC), "LAN":int(LAN_FRC), "NAV":int(NAV_FRC), "OBS":int(OBS_FRC)}
FRC_DICT = {"SEP":SEP_FRC, "ALI":ALI_FRC, "ATT":ATT_FRC, "LAN":LAN_FRC, "NAV":NAV_FRC, "OBS":OBS_FRC}

# 障害物の有無
IS_OBSTACLE = True

# 交差点の有無
IS_INTER = False

# 仮想リーダーの有無
VIRTUAL_LEADER = True

# Flockingアルゴリズムによる縦方向制御
IS_LONG_FLOCK = False

# 流体力学による障害物回避
IS_OBS_POTENTIAL = True


# マルチプロセシング・NSGA-iiiの設定
N_CPU = cpu_count() # CPUの数(自動で設定)
N_LOOP = 1 # NSGA-iiiの反復回数
N_POP = 10 # 個体群の数(population)
N_GEN = 20 # 世代数(generation)

# ヒートマップ
HEAT_V_MIN = 4.0
HEAT_V_MAX = 12.0
HEAT_AGENT_MIN = 3
HEAT_AGENT_MAX = 9 
HEAT_X_NUM = 5 # x軸の要素数
HEAT_Y_NUM = 7 # y軸の要素数

# Pymooの設定
IS_SINGLE_OBJ = False # 単目的最適化かどうか # 世代ごとの目的値の推移の評価用
[OBJ_CORR, OBJ_COLL, OBJ_OBS, OBJ_DIS, OBJ_VEL] = [False, True, True, True, False]
OBJ_LIST = [OBJ_CORR, OBJ_COLL, OBJ_OBS, OBJ_DIS, OBJ_VEL] # 目的関数のリスト
OBJ_DICT = {"CORR":OBJ_CORR, "COLL":OBJ_COLL, "OBS":OBJ_OBS, "DIS":OBJ_DIS, "VEL":OBJ_VEL}
OBJ_LABEL_DICT = {"CORR":"obj_corr", "COLL":"obj_coll", "OBS":"obj_obs", "DIS":"obj_dis", "VEL":"obj_vel"}
if IS_SINGLE_OBJ: # 単目的最適化の場合
    N_OBJ = 1
else: # 多目的最適化の場合
    N_OBJ = OBJ_LIST.count(True)  # 目的関数の数(objective)
XL, XU = set.set_parameter_limit(lower=L_LIM_DICT, upper=U_LIM_DICT, frc_dict=FRC_DICT)
N_VAR = len(XL) # Flockingパラメータの数
N_PAR = 14 # Das-DennisのPartitionの数
N_CNST = 0 # 制約条件の数

# 選択パラメータ
N_ROW = 3 # パラメータセットの行番号

