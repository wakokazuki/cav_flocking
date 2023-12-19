from datetime import datetime
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import starmap_parallelized_eval
from pymoo.factory import get_reference_directions, get_termination
import multiprocessing
import numpy as np
import pandas as pd
import random 
import csv
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameters import params
from main import run
from settings import set, show

# Config.show_compile_hint = False

class MyProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=params.N_VAR, n_obj=params.N_OBJ, n_constr=params.N_CNST, xl=params.XL, xu=params.XU, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        eval_list = [] 
        objective_dict = run.main_program(parameter=x, max_vel=params.V_MAX, agent_num=params.N_FOLLOWER, show_flag=False, return_flag=True)
        # 目的関数値のリストを作成
        # 単目的最適化の場合
        if params.IS_SINGLE_OBJ:
            single_obj_value = objective_dict["COLL"] + objective_dict["OBS"] + objective_dict["DIS"]
            eval_list.append(single_obj_value)
        # 多目的最適化の場合
        else:
            for keys in params.OBJ_DICT.keys():
                if params.OBJ_DICT[keys]:
                    eval_list.append(objective_dict[keys])
        out["F"] = eval_list # Maximize -> Minimize


if __name__ == "__main__":
    # 最適化手法のラベル
    method_label = "ctaea"
    # 開始時刻
    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M") 
    # CSVファイルのラベルのリストを作成
    # 単目的最適化の場合
    if params.IS_SINGLE_OBJ:
        columns_rabel = set.set_column(frc_dict=params.FRC_DICT, 
                                       label_dict=params.LABEL_DICT, 
                                       obj_dict=params.OBJ_DICT, 
                                       obj_label_dict=params.OBJ_LABEL_DICT,
                                       single_obj=True)
    else: # 多目的最適化の場合
        columns_rabel = set.set_column(frc_dict=params.FRC_DICT, 
                                       label_dict=params.LABEL_DICT,
                                       obj_dict=params.OBJ_DICT, 
                                       obj_label_dict=params.OBJ_LABEL_DICT,
                                       single_obj=False)
    
    # マルチプロセシング
    _ref_dirs = get_reference_directions("das-dennis", n_dim=params.N_OBJ, n_partitions=params.N_PAR)
    loop_num = params.N_LOOP
    pop_num = params.N_POP
    gen_num = params.N_GEN  
    for loop in range(loop_num):
        print(f"Loop:{loop+1}")
        _seed = random.randint(0, 10000)
        n_process = params.N_CPU 
        pool = multiprocessing.Pool(n_process)
        _problem = MyProblem(runner=pool.starmap, func_eval=starmap_parallelized_eval)
        _algorithm = CTAEA(ref_dirs=_ref_dirs)
        _termination = get_termination("n_gen", gen_num)
        res = minimize(problem=_problem,
                       algorithm=_algorithm,
                       termination=_termination,
                       seed=_seed,
                       verbose=True,
                       save_history=params.IS_SINGLE_OBJ)
        print(f"Time:{res.exec_time}")
        
        # 目的関数値の抽出
        obj_value = res.F
        param_value = res.X

        # 収束性の確認
        if params.IS_SINGLE_OBJ:
            opt_history = np.array([e.opt[0].F[0] for e in res.history])
            show.objective_value_convergence(value=opt_history, date_time=date_time, loop_num=loop, method_label=method_label)
            with open(f"../out/convergence/csv/convergence_{date_time}_loop{loop:02}_obs{int(params.IS_OBSTACLE):02}_vl{int(params.VIRTUAL_LEADER):02}_sp{int(params.V_MAX):02}_num{int(params.N_FOLLOWER):02}_pop{pop_num}_gen{gen_num}_{method_label}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(opt_history)
            
        # CSVファイルを保存
        if params.IS_SINGLE_OBJ: # 単目的最適化の場合
            pareto_solution = np.append(param_value, obj_value) # 1次元配列
        else: # 多目的最適化の場合
            pareto_solution = np.concatenate([param_value, obj_value], axis=1)
        df = pd.DataFrame(pareto_solution.T, index=columns_rabel).T
        df.to_csv(f"../out/csv/{date_time}_loop{loop:02}_obs{int(params.IS_OBSTACLE):02}_vl{int(params.VIRTUAL_LEADER):02}_pop{pop_num}_gen{gen_num}_{method_label}.csv")