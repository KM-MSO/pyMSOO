import numpy as np
import pandas as pd

from pyMSOO.MFEA.model import SM_MFEA
from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils import DimensionAwareStrategy

from pyMSOO.utils.EA import * 

from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 
from pyMSOO.MFEA.benchmark.continous.utils import Individual_func
from pyMSOO.utils.LoadSaveModel.load_utils import loadModel


class func():
    
    def __init__(self, n, m, a: np.ndarray, L: List[float], U: List[float], c: np.ndarray, idx_n):
        assert a.shape[0] == len(L) == len (U) == m
        assert a.shape[1] == len(c) == n == len(idx_n)
        self.a = a
        self.L = L
        self.U = U
        self.c = c
        self.dim = n
        self.idx_n = idx_n
    
    def check_valid(self, x):
        mat_mul = self.a[:, self.idx_n] @ x[self.idx_n]
        w_L = mat_mul - self.L
        w_U = self.U - mat_mul
        if (np.all(w_L >= 0) and np.all(w_U >= 0)):
            return True, None, None, None
        else:
            return False, mat_mul, self.L, self.U

    @staticmethod
    @jit(nopython=True)
    def get_f(x, a, L, U, c, idx_n):
        mat_mul = a[:, idx_n] @ x[idx_n]
        w_L = mat_mul - L
        w_U = U - mat_mul
        if not (np.all(w_L >= 0) and np.all(w_U >= 0)):
            sum_arr = np.where(w_L < 0, - w_L, 0) + np.where(w_U < 0, - w_U, 0)
            return 1e10 + 1e20 * np.sum(sum_arr)
            # return c[idx_n] @ x[idx_n] * ( 1 + 1e50 * np.sum(sum_arr))
        else:
            return c[idx_n] @ x[idx_n]

    def __call__(self, x: np.ndarray):
        return self.__class__.get_f(x, self.a, self.L, self.U, self.c, self.idx_n )
    
def get_data(idx):
    with open(f"LP_data/LP{idx}.txt", "r") as f:
        content = f.read()
    lines = [l.strip().split(' ') for l in content.split('\n')]

    n, m = np.array(lines[0], dtype=int)
    c = np.array(lines[1], dtype=float)

    L_U = np.array([lines[2*i + 2] for i in range(m)], dtype=float)
    L = L_U[:, 0]
    U = L_U[:, 1]

    a = np.array([lines[2*i + 3] for i in range(m)], dtype= float)

    print(c.shape, L.shape, U.shape, a.shape)
    return n, m, c, L, U, a

def get_tasks(idx, nb_tasks= 10, per_C=0.3):
    n, m, c, L, U, a = get_data(idx)

    obj_func = func(n, m, a, L, U, c, np.arange(n))
    obj_func.name = "obj"

    others_func = []
    for _ in range(nb_tasks):
        # _n, _m = np.random.randint(n), np.random.randint(m)
        _n, _m = int(n * 1), int(m * per_C)
        
        idx_n = np.arange(n)
        # idx_n = np.random.choice(_n, size=_n, replace= False)
        idx_m = np.random.choice(_m, size=_m, replace= False)
        if np.random.rand() < 0:
            others_func.append(func(_n, _m, a[idx_m][:, idx_n], L[idx_m], U[idx_m], c[idx_n], idx_n))
        else:
            others_func.append(func(_n, _m, 
                                    a[idx_m][:, idx_n],
                                    # a[idx_m][:, idx_n] + np.random.rand(len(idx_m), len(idx_n)), 
                                    L[idx_m], U[idx_m],
                                    # L[idx_m] + np.random.rand(len(idx_m)), 
                                    # U[idx_m] + np.random.rand(len(idx_m)), 
                                    c[idx_n] + np.random.rand(len(idx_n)),
                                    idx_n))


    return [obj_func] + others_func, Individual_func

def run(DaS, idx_data, nb_tasks, per_C, nb_run= 3):
    tasks, IndClass = get_tasks(idx_data, nb_tasks, per_C)
    n = tasks[0].dim

    SM_SBX = MultiTimeModel(model= SM_MFEA)
    # SM_SBX = SM_MFEA.model()
    SM_SBX.compile(
        IndClass= IndClass,
        tasks= tasks,
        crossover= SBX_Crossover(nc = 2),
        mutation= PolynomialMutation(nm = 7, pm= 1/n),
        # selection= ElitismSelection(random_percent= 0.05), 
        dimension_strategy= DimensionAwareStrategy.DaS_strategy(eta= 3) if DaS else DimensionAwareStrategy.NoDaS()
    )
    SM_SBX.fit(
        nb_generations= max(n * 15, 2000), nb_inds_each_task= n * 2, nb_inds_min= n * 1,
        lr = 0.1, mu=0.1,
        evaluate_initial_skillFactor= True
    )
    SM_SBX.run(
        nb_run= nb_run,
        save_path= f'./THESIS_RESULTS/LP{idx_data}_{DaS}.mso'
    )

if __name__ == '__main__':
    import sys
    mode, idx_data, nb_tasks, per_C = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    run(mode, idx_data, nb_tasks, per_C)