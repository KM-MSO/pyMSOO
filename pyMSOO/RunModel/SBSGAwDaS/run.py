from pyMSOO.MFEA.model import MFEA_base, SM_MFEA, LSA21, SBSGA
from pyMSOO.MFEA.competitionModel import SM_MFEA_Competition

from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils.Search import * 
from pyMSOO.utils.DimensionAwareStrategy import DaS_strategy 
from pyMSOO.MFEA.benchmark.continous import *
from pyMSOO.utils.MultiRun.RunMultiTime import * 

from pyMSOO.utils.EA import * 
from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark
from pyMSOO.MFEA.benchmark.continous.funcs import * 

from pyMSOO.utils.MultiRun.RunMultiTime import * 
from pyMSOO.utils.MultiRun.RunMultiBenchmark import * 

from pyMSOO.utils.LoadSaveModel.load_utils import loadModel

from pyMSOO.utils.numba_utils import *

from pyMSOO.RunModel.config import benchmark_cfg

import argparse 
import yaml 


parser = argparse.ArgumentParser(description='MaTGA no DaS Running')

# t, ic = CEC17_benchmark.get_10tasks_benchmark()

# ls_benchmark = [t]
# ls_IndClass = [ic]
# name_benchmark = ["cec17"]

def add_argument():
    global parser 
    default_params = {} 

    with open("cfg.yaml", encoding='utf-8') as file:
        default_params.update(yaml.load(file, Loader= yaml.Loader))

    for key, value in default_params.items():
        parser.add_argument(
            f"--{key}", 
            default= value,
            type= type(value), 
        )

add_argument()



def main():

    # process args 
    args = parser.parse_args()

    ls_benchmark, ls_IndClass, name_benchmark = getattr(benchmark_cfg.config, f'{args.name_benchmark}_{args.number_tasks}')() 

    # process custom id run 
    if args.ls_id_run != "1-1-1":
        print(args.ls_id_run)
        ls_id_tasks = args.ls_id_run.split("-") 
        ls_benchmark = [ls_benchmark[int(i)] for i in ls_id_tasks]
        ls_IndClass = [ls_IndClass[int(i)] for i in ls_id_tasks]
        name_benchmark = [name_benchmark[int(i)] for i in ls_id_tasks]
    

    sbsgaModel = MultiBenchmark(
        ls_benchmark= ls_benchmark,
        name_benchmark= name_benchmark,
        ls_IndClass= ls_IndClass,
        model= SBSGA
    )

    sbsgaModel.compile( 
        crossover= SBX_Crossover(nc = args.nc),
        mutation= PolynomialMutation(nm = args.nm),
        dimension_strategy= DaS_strategy(eta= args.eta),
        selection = ElitismSelection()
    )
    sbsgaModel.fit(
        nb_generations= args.nb_generations, 
        rmp = args.rmp, 
        nb_inds_each_task= args.nb_inds_each_task,
        bound_pop = [0, 1],
        evaluate_initial_skillFactor= args.evaluate_initial_skillFactor 
    )
    a = sbsgaModel.run(
        nb_run= args.nb_run,     
        save_path= args.save_path
    )



if __name__ == '__main__':
    main()