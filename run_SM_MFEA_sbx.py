from pyMSOO.MFEA.model import MFEA_base, SM_MFEA, LSA21
from pyMSOO.MFEA.competitionModel import SM_MFEA_Competition
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
from pyMSOO.utils.DimensionAwareStrategy import DaS_strategy, NoDaS
from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 

from pyMSOO.utils.LoadSaveModel.load_utils import loadModel, loadModelFromTxt
from pyMSOO.utils.LoadSaveModel.save_utils import export_history2txt

from pyMSOO.utils.numba_utils import *
from pyMSOO.utils.Compare.compareModel import CompareModel

import pandas as pd 
import numpy as np 
import os 

ls_benchmark = []
ls_IndClass = []
ls_tasks = [10]
name_benchmark = [] 

for i in ls_tasks:
    # t, ic = WCCI22_benchmark.get_complex_benchmark(i)
    t, ic = WCCI22_benchmark.get_50tasks_benchmark(i)
    ls_benchmark.append(t)
    ls_IndClass.append(ic)
    name_benchmark.append(str(i))

smpModel = MultiBenchmark(
    ls_benchmark= ls_benchmark,
    name_benchmark= name_benchmark,
    ls_IndClass= ls_IndClass,
    model= SM_MFEA
)

smpModel.compile( 
        crossover= SBX_Crossover(nc = 2),
        mutation= PolynomialMutation(nm = 5, pm=1),
        dimension_strategy= NoDaS(),
        search = DifferentialEvolution.L_SHADE(p_ontop= 0.11, len_mem= 30),
        selection = ElitismSelection(random_percent= 0.0)
)
smpModel.fit(
        nb_generations= 500, nb_inds_each_task= 100, nb_inds_min= 20,
        lr = 0.1 ,mu= 0.1,
        evaluate_initial_skillFactor= True  
)
a = smpModel.run(
    nb_run=15,     
    save_path= './RESULTS/result/GECCO20/check/SMP_MFEA_SBX/'
)