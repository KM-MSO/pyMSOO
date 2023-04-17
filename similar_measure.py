import os 
import sys 
import numpy as np 
import pandas as pd
import scipy 
import inspect 
import pickle 

from pyMSOO.MFEA.model import MFEA_base, SM_MFEA
from pyMSOO.MFEA.competitionModel import LSA21, SM_MFEA_Competition 
from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils.Search import * 
from pyMSOO.MFEA.benchmark.continous import *
from pyMSOO.utils.MultiRun.RunMultiTime import * 

from pyMSOO.utils.EA import * 
from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark
from pyMSOO.MFEA.benchmark.continous.GECCO20 import GECCO20_benchmark_50tasks
from pyMSOO.MFEA.benchmark.continous.utils import Individual_func 
from pyMSOO.MFEA.benchmark.continous.funcs import * 

from pyMSOO.utils.EA import Population

from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 

from pyMSOO.utils.LoadSaveModel.LoadModel import loadModel

from pyMSOO.utils.numba_utils import *


tasks, IndClass = CEC17_benchmark.get_10tasks_benchmark()

uss_population = np.random.rand(1000000, 50)

result= np.zeros(shape=(10,10))
for task1 in range(1, 11):
    for task2 in range(1, 11):
        # task1, task2 = 1,6
        task1 -= 1
        task2 -= 1

        if True:
            cost_task1 = [] 
            for i in uss_population: 
                cost_task1.append(tasks[task1](i))
            cost_task1 = np.array(cost_task1)

        if True:
            cost_task2 = [] 
            for i in uss_population: 
                cost_task2.append(tasks[task2](i))
            cost_task2 = np.array(cost_task2)
        fac_rank1 = np.argsort(np.argsort(cost_task1))
        fac_rank2 = np.argsort(np.argsort(cost_task2))
        cov = np.cov(np.stack((fac_rank1, fac_rank2), axis= 0))[0][1]
        std1 = np.std(fac_rank1)
        std2 = np.std(fac_rank2)
        R_s = cov / (std1 * std2)
        result[task1][task2]= R_s

print(result)
np.savetxt("foo.csv", np.round(result, 3), delimiter=",")