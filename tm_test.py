from pyMSOO.MFEA.model import MFEA_base, SM_MFEA, LSA21
from pyMSOO.MFEA.competitionModel import SM_MFEA_Competition, MFEA_Multiparent
from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils.Search import * 
from pyMSOO.MFEA.benchmark.continous import *
from pyMSOO.utils.MultiRun.RunMultiTime import * 

from pyMSOO.utils.EA import * 
from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark
from pyMSOO.MFEA.benchmark.continous.funcs import * 

from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 
from pyMSOO.utils.Compare.compareModel import CompareModel

from pyMSOO.utils.LoadSaveModel.load_utils import loadModel, loadModelFromTxt

from pyMSOO.utils.numba_utils import *

from pyMSOO.utils.Compare.utils import render_cec_17

# global_optima_all = []

# for ID in range(1, 11):
#     ls_tasks = WCCI22_benchmark.get_50tasks_benchmark(ID)[0] 

#     global_optima = [] 

#     for task in ls_tasks:
#         global_optima.append(task.global_optimal)

#     import numpy as np 

#     np.save(f"global_optima_WCCI22_ID_{ID}.npy",global_optima)
#     global_optima_all.append(global_optima)

# np.save("global_optimal_all_WCCI22.npy", global_optima_all)


