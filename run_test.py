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
from pyMSOO.MFEA.benchmark.continous.utils import Individual_func 
from pyMSOO.MFEA.benchmark.continous.funcs import * 

from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 
from pyMSOO.utils.LoadSaveModel.load_utils import loadModel 
from pyMSOO.utils.numba_utils import *


# t, ic = CEC17_benchmark.get_10tasks_benchmark()

# ls_benchmark = [t]
# ls_IndClass = [ic]
# name_benchmark = ["cec17"]

ls_benchmark = []
ls_IndClass = []
ls_tasks = [1]
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
    model= SM_MFEA_Competition
)

smpModel.compile(

    # crossover= SBX_LSA21(nc=2, k=7),
    crossover = DaS_SBX_Crossover(nc=2, eta=3, conf_thres= 1),
    selection= ElitismSelection(random_percent= 0.),
    mutation= NoMutation(),
    search = LSHADE_LSA21(len_mem=30, p_ontop=0.11),
    attr_tasks = ['crossover', 'mutation', 'search'],
)
smpModel.fit(
    nb_generations= 1000, nb_inds_each_task= 100, nb_inds_min= 20,
    lr = 0.1, p_const_intra= 0., prob_search = 0., lc_nums = 200,
    nb_epochs_stop= 1000, swap_po= False,
    evaluate_initial_skillFactor= True
)
a = smpModel.run(
    nb_run= 5,
    save_path= './RESULTS/SMP_SBX_DSCG/'
)

# smpModel = MultiBenchmark(
#     ls_benchmark= ls_benchmark,
#     name_benchmark= name_benchmark,
#     ls_IndClass= ls_IndClass,
#     model= SM_MFEA
# )
# smpModel.compile( 
#     # crossover = KL_SBXCrossover(nc= 2, u= 0.001, conf_thres= 1),
#     # crossover = SBX_Crossover(nc = 2),
#     crossover = DaS_SBX_Crossover(nc=2, eta= 3, conf_thres= 0.6),
#     mutation = PolynomialMutation(nm = 5, pm= 1),
#     selection= ElitismSelection(random_percent= 0.1),
#     search= L_SHADE(len_mem= 15),
#     attr_tasks = ['crossover', 'mutation', 'search'],
# )
# smpModel.fit(
#     nb_generations= 1000, nb_inds_each_task= 100, nb_inds_min= 20,
#     lr = 0.1, p_const_intra= 0., prob_search = 0., lc_nums = 200,
#     nb_epochs_stop= 1000, swap_po= False,
#     evaluate_initial_skillFactor= True
# )
# a = smpModel.run(
#     nb_run= 1,     
#     save_path= './RESULTS/'
# )

