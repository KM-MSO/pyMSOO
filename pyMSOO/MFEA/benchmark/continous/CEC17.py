from .utils import Individual_func, AbstractFunc
from .funcs import Sphere, Weierstrass, Rosenbrock, Ackley, Schwefel, Griewank, Rastrigin

from typing import Tuple, Type, List
import os
from scipy.io import loadmat
import numpy as np 
path = os.path.dirname(os.path.realpath(__file__))

class CEC17_benchmark():
    dim = 50
    task_size = 2

    def get_10_tasks_benchmark_ver_quite_complicate(fix= False) -> Tuple[List[AbstractFunc], Type[Individual_func]]: 
        tasks = [
        Sphere(     50,shift= [-80.0,0.0,-80.0,0.0,-80.0,0.0,-80.0,0.0,-80.0,0.0] * 5,    bound= [-100, 100]),   # 0
        Sphere(     50,shift= [80.0,-80.0,80.0,-80.0,80.0,-80.0,80.0,-80.0,80.0,-80.0] * 5,   bound= [-100, 100]),  # 80
        Sphere(     50,shift= [40.0,80.0,40.0,80.0,40.0,80.0,40.0,80.0,40.0,80.0] * 5,  bound= [-100, 100]), # -80
        Weierstrass(50,shift= np.array([0.3,-0.3,0.3,-0.3,0.3,-0.3,0.3,-0.3,0.3,-0.3] * 5), bound= [-0.5, 0.5]), # -0.4
        Rosenbrock( 50,shift= [-41.0,-1.0,-41.0,-1.0,-41.0,-41.0,39.0,-41.0,39.0,-41.0] * 5,   bound= [-50, 50]),# 0
        Ackley(     50,shift= [40.0,-40.0,40.0,-40.0,40.0,0.0,-40.0,0.0,-40.0,0.0] * 5,   bound= [-50, 50]),    # 40
        Weierstrass(50,shift= [0.2,-0.3,-0.4,0.4,0.4,0.0,0.2,-0.4,0.3,-0.4] * 5, bound= [-0.5, 0.5]), # -0.4
        Schwefel(   50,shift= 0,    bound= [-500, 500], fixed = fix), # 420.9687
        Griewank(   50,shift= [60.0,0.0,80.0,-60.0,40.0,-60.0,60.0,80.0,-80.0,-80.0] * 5, bound= [-100, 100]), # -80, 80
        Rastrigin(  50,shift= [-40.0,-40.0,20.0,0.0,30.0,-30.0,20.0,-40.0,20.0,0.0] * 5, bound= [-50, 50]),# -40, 40
        ]
        return tasks, Individual_func
    
    def get_10_tasks_benchmark_ver_max_complicate(fix= False) -> Tuple[List[AbstractFunc], Type[Individual_func]]: 
        tasks = [
        Sphere(     50,shift= [-80.0,0.0,-80.0,0.0,-80.0,0.0,-80.0,0.0,-80.0,0.0] * 5,    bound= [-100, 100]),   # 0
        Sphere(     50,shift= [80.0,-80.0,80.0,-80.0,80.0,-80.0,80.0,-80.0,80.0,-80.0] * 5,   bound= [-100, 100]),  # 80
        Sphere(     50,shift= [40.0,80.0,40.0,80.0,40.0,80.0,40.0,80.0,40.0,80.0] * 5,  bound= [-100, 100]), # -80
        Weierstrass(50,shift= np.array([0.3,-0.3,0.3,-0.3,0.3,-0.3,0.3,-0.3,0.3,-0.3] * 5), bound= [-0.5, 0.5]), # -0.4
        Rosenbrock( 50,shift= [-41.0,-41.0,19.0,-1.0,-41.0,-41.0,-41.0,-31.0,29.0,39.0] * 5,   bound= [-50, 50]),# 0
        Ackley(     50,shift= [40.0,40.0,30.0,-40.0,30.0,40.0,40.0,-30.0,-40.0,0.0] * 5,   bound= [-50, 50]),    # 40
        Weierstrass(50,shift= [0.2,-0.3,-0.4,0.4,0.4,0.0,0.2,-0.4,0.3,-0.4] * 5, bound= [-0.5, 0.5]), # -0.4
        Schwefel(   50,shift= 0,    bound= [-500, 500], fixed = fix), # 420.9687
        Griewank(   50,shift= [60.0,0.0,80.0,-60.0,40.0,-60.0,60.0,80.0,-80.0,-80.0] * 5, bound= [-100, 100]), # -80, 80
        Rastrigin(  50,shift= [-40.0,-40.0,20.0,0.0,30.0,-30.0,20.0,-40.0,20.0,0.0] * 5, bound= [-50, 50]),# -40, 40
        ]
        return tasks, Individual_func


    def get_10tasks_benchmark(fix = False)-> Tuple[List[AbstractFunc], Type[Individual_func]]:
        tasks = [
        Sphere(     50,shift= 0,    bound= [-100, 100]),   # 0
        Sphere(     50,shift= 80,   bound= [-100, 100]),  # 80
        Sphere(     50,shift= -80,  bound= [-100, 100]), # -80
        Weierstrass(25,shift= -0.4, bound= [-0.5, 0.5]), # -0.4
        Rosenbrock( 50,shift= -1,   bound= [-50, 50]),# 0
        Ackley(     50,shift= 40,   bound= [-50, 50]),    # 40
        Weierstrass(50,shift= -0.4, bound= [-0.5, 0.5]), # -0.4
        Schwefel(   50,shift= 0,    bound= [-500, 500], fixed = fix), # 420.9687
        Griewank(   50,shift= [-80, 80], bound= [-100, 100]), # -80, 80
        Rastrigin(  50,shift= [40, -40], bound= [-50, 50]),# -40, 40
        ]
        return tasks, Individual_func


    def get_2tasks_benchmark(ID)-> Tuple[List[AbstractFunc], Type[Individual_func]]:
        #TODO
        tasks = []

        if ID == 1:
            ci_h = loadmat(path + "/__references__/CEC17/Tasks/CI_H.mat")
            tasks.append(
                Griewank(
                    dim= 50,
                    shift = ci_h['GO_Task1'],
                    rotation_matrix= ci_h['Rotation_Task1'],
                    bound= (-100, 100)
                )
            )
            tasks.append(
                Rastrigin(
                    dim= 50,
                    shift= ci_h['GO_Task2'],
                    rotation_matrix= ci_h['Rotation_Task2'],
                    bound= (-50, 50)
                )
            )
        elif ID == 2:
            ci_m = loadmat(path + "/__references__/CEC17/Tasks/CI_M.mat")
            tasks.append(
                Ackley(
                    dim= 50,
                    shift= ci_m['GO_Task1'],
                    rotation_matrix= ci_m['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Rastrigin(
                    dim= 50,
                    shift= ci_m['GO_Task2'],
                    rotation_matrix= ci_m['Rotation_Task2'],
                    bound= (-50, 50)
                )
            )
        elif ID == 3:
            ci_l = loadmat(path + "/__references__/CEC17/Tasks/CI_L.mat")
            tasks.append(
                Ackley(
                    dim= 50,
                    shift= ci_l['GO_Task1'],
                    rotation_matrix= ci_l['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Schwefel(
                    dim= 50,
                    bound= (-500, 500)
                )
            )
        elif ID == 4:
            pi_h = loadmat(path + "/__references__/CEC17/Tasks/PI_H.mat")
            tasks.append(
                Rastrigin(
                    dim= 50, 
                    shift= pi_h['GO_Task1'],
                    rotation_matrix= pi_h['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Sphere(
                    dim= 50, 
                    shift = pi_h['GO_Task2'],
                    bound= (-100, 100)
                )
            )
        elif ID == 5:
            pi_m = loadmat(path + "/__references__/CEC17/Tasks/PI_M.mat")
            tasks.append(
                Ackley(
                    dim= 50, 
                    shift= pi_m['GO_Task1'],
                    rotation_matrix= pi_m['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Rosenbrock(
                    dim= 50, 
                    bound= (-50, 50)
                )
            )
        elif ID == 6:
            pi_l = loadmat(path + "/__references__/CEC17/Tasks/PI_L.mat")
            tasks.append(
                Ackley(
                    dim= 50, 
                    shift= pi_l['GO_Task1'],
                    rotation_matrix= pi_l['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Weierstrass(
                    dim= 25, 
                    shift = pi_l['GO_Task2'],
                    rotation_matrix= pi_l['Rotation_Task2'],
                    bound= (-0.5, 0.5)
                )
            )
        elif ID == 7:
            ni_h = loadmat(path + "/__references__/CEC17/Tasks/NI_H.mat")
            tasks.append(
                Rosenbrock(
                    dim= 50, 
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Rastrigin(
                    dim= 50, 
                    shift = ni_h['GO_Task2'],
                    rotation_matrix= ni_h['Rotation_Task2'],
                    bound= (-50, 50)
                )
            )
        elif ID == 8:
            ni_m = loadmat(path + "/__references__/CEC17/Tasks/NI_M.mat")
            tasks.append(
                Griewank(
                    dim= 50, 
                    shift= ni_m['GO_Task1'],
                    rotation_matrix= ni_m['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Weierstrass(
                    dim= 50, 
                    shift = ni_m['GO_Task2'],
                    rotation_matrix= ni_m['Rotation_Task2'],
                    bound= (-0.5, 0.5)
                )
            )
        elif ID == 9:
            ni_l = loadmat(path + "/__references__/CEC17/Tasks/NI_L.mat")
            tasks.append(
                Rastrigin(
                    dim= 50, 
                    shift= ni_l['GO_Task1'],
                    rotation_matrix= ni_l['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Schwefel(
                    dim= 50, 
                    bound= (-500, 500)
                )
            )
        else:
            raise ValueError('ID must be an integer from 1 to 9, not ' + ID)
        return tasks, Individual_func
