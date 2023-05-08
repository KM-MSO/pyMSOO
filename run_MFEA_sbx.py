from pyMSOO.MFEA.model import MFEA_base, SM_MFEA, LSA21
from pyMSOO.MFEA.competitionModel import SM_MFEA_Competition

from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils.Search import * 
from pyMSOO.utils.DimensionAwareStrategy import DaS_strategy, NoDaS
from pyMSOO.MFEA.benchmark.continous import *
from pyMSOO.utils.MultiRun.RunMultiTime import * 

from pyMSOO.utils.EA import * 
from pyMSOO.MFEA.benchmark.continous.funcs import * 

from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 

from pyMSOO.utils.numba_utils import *

from pyMSOO.RunModel.config import benchmark_cfg
# gecco20
ls_benchmark = []
ls_IndClass = []
name_benchmark = []
path = './RESULTS/result/GECCO20/SMP_v2/'

for i in [2]:
    t, ic = WCCI22_benchmark.get_50tasks_benchmark(i)
    ls_benchmark.append(t)
    ls_IndClass.append(ic)
    name_benchmark.append(i)


# path = './RESULTS/result/CEC17_10tasks/SMP_KL/'

mfeaModel = MultiBenchmark(
    ls_benchmark= ls_benchmark,
    name_benchmark= name_benchmark,
    ls_IndClass= ls_IndClass,
    model= MFEA_base
)
mfeaModel.compile( 
    # crossover = KL_SBXCrossover(nc= 2, u= 0.001, conf_thres= 1),
    crossover= SBX_Crossover(nc = 2),
    mutation= NoMutation(),
    selection= ElitismSelection()
)
mfeaModel.fit(
    nb_generations = 1000, rmp = 0.3, nb_inds_each_task= 100, 
    bound_pop= [0, 1], evaluate_initial_skillFactor= True
)
a = mfeaModel.run(
    nb_run= 5,     
    save_path= path
)