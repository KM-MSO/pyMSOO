from ...utils.EA import * 
from ...utils import Crossover, Mutation, Selection, Search
from pyMSOO.utils.EA import AbstractTask 
from ..model import AbstractModel 
import numpy as np 
from ...utils.numba_utils import * 
import matplotlib.pyplot as plt 

import time 
from collections import defaultdict

class model(AbstractModel.model):

    class rmp_lsa: 
        def __init__(self, nb_tasks, default_rmp= 0.5) -> None:
            self.default_rmp = default_rmp
            self.C = 0.02 
            self.best_partner = None 
            self.nb_tasks = nb_tasks 

            self.rmp = np.zeros(shape=(self.nb_tasks, self.nb_tasks)) + self.default_rmp 
            self.best_partner = np.zeros(shape= (self.nb_tasks), dtype= np.int64)  - 1 

            self.s_rmp = np.empty(shape= (self.nb_tasks, self.nb_tasks, 0)).tolist() 
            self.diff_f_inter_x = np.empty(shape=(self.nb_tasks, self.nb_tasks, 0)).tolist() 
        
        def get_rmp(self):
            return self.rmp.copy()

        def update(self, *args, **kwargs): 
            for task in range(self.nb_tasks):
                max_rmp = 0 
                self.best_partner[task] = -1 

                for task2 in range(self.nb_tasks):
                    if task2 == task: 
                        continue 
                    good_mean = 0 
                    if len(self.s_rmp[task][task2]) > 0: 
                        sum_diff = np.sum(np.array(self.diff_f_inter_x[task][task2]))
                        
                        w = np.array(self.diff_f_inter_x[task][task2]) / sum_diff 

                        val1 = np.sum(w * np.array(self.s_rmp[task][task2]) **2)
                        val2 = np.sum(w * np.array(self.s_rmp[task][task2])) 

                        good_mean = val1 / val2 

                        if (good_mean > max_rmp):
                            max_rmp = good_mean 
                            self.best_partner[task] = task2                 
                        pass 
                    
                    if good_mean > 0: 
                        c1 = 1.0 
                    else: 
                        c1 = 1.0 - self.C 
                    
                    #TM-FIXME: Review: lieu cach update rmp da hop ly ? 
                    self.rmp[task][task2] = c1 * self.rmp[task][task2] + self.C * good_mean 
                    self.rmp[task][task2] = np.clip(self.rmp[task][task2], 0.01, 1)
            self.s_rmp = np.empty(shape= (self.nb_tasks, self.nb_tasks, 0)).tolist() 
            self.diff_f_inter_x = np.empty(shape=(self.nb_tasks, self.nb_tasks, 0)).tolist() 
    
    def __init__(self, seed= None, percent_print= 1,*args, **kwargs):
        super().__init__(seed, percent_print, *args, **kwargs)
        
    def compile(self, 
                IndClass: Type[Individual], 
                tasks: list[AbstractTask], 
                crossover: Crossover.AbstractCrossover, 
                mutation: Mutation.AbstractMutation,
                search: Search.DifferentialEvolution.LSHADE_LSA21, 
                local_search: Search.LocalSearch_DSCG, 
                # TM-TODO: adding multiparent crossover
                selection: Selection.ElitismSelection, 
                *args, **kwargs,
                ):
        super().compile(
            IndClass, 
            tasks, 
            crossover= crossover, 
            mutation= mutation,
            selection= selection, 
            *args, **kwargs
        )

        self.search = search
        self.search.getInforTasks(
            IndClass= IndClass, 
            tasks= tasks, 
            seed = self.seed,
        )
        self.rmp = self.rmp_lsa(len(self.tasks))

    def render_rmp(self, shape= None, title= None, 
                   figsize= None, dpi= 100, step= 1, 
                   label_loc= 'lower center', label_shape = None):
        
        # ls_color = ["#OB2559", "#615C84", "#FFD8B9", "#F2ABB3", "#B1CAE9", "#FBB579", "#EEDCCE", "#03417F","#5883AD","#FEAFA2"]
        if title is None: 
            title= self.__class__.__name__ 
        
        if shape is None: 
            shape = (int(np.ceil(len(self.tasks) / 3)), 3)
        else : 
            assert shape[0] * shape[1] >= len(self.tasks)

        if label_shape is None: 
            label_shape= (1, len(self.tasks))
        else: 
            assert label_shape[0] * label_shape[1] >= len(self.tasks) 
        
        if figsize is None: 
            figsize= (shape[1] * 6, shape[0] * 5) 
        
        fig = plt.figure(figsize= figsize, dpi = dpi) 
        fig.suptitle(title, size= 15) 
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        his_rmp = np.copy(self.history_rmp)
        y_lim = (-0.1, 1.1)

        for idx_task, task in enumerate(self.tasks): 
            for idx_task2, _ in enumerate(self.tasks):
                # if idx_task2 == idx_task: 
                #     continue
                fig.axes[idx_task].plot(
                    np.append(np.arange(0, len(his_rmp), step), np.array([len(his_rmp) - 1])),
                    his_rmp[
                        np.append(np.arange(0, len(his_rmp), step), np.array([len(his_rmp) - 1])), 
                        idx_task, idx_task2] ,
                    label = 'Task ' + str(idx_task2),
                    # color= ls_color[idx_task2]
                    
                )

        lines, labels = fig.axes[0].get_legend_handles_labels() 

        fig.tight_layout()
        fig.legend(lines, labels, loc= label_loc, ncol= label_shape[1])

        plt.show() 
        plt.savefig("hello.png")
        pass 

    
    def fit(self, nb_generations: int, nb_inds_each_task: int, 
                nb_inds_min= None, evaluate_initial_skillFactor= True,
                *args, **kwargs):
        
        super().fit(*args, **kwargs)

        if nb_inds_min is None: 
            nb_inds_min = nb_inds_each_task 
        else: 
            assert nb_inds_each_task > nb_inds_min 
        
        nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)

        self.history_rmp = [] 

        # Init population 
        population = Population(
            self.IndClass, 
            nb_inds_tasks= [nb_inds_each_task] * len(self.tasks), 
            dim= self.dim_uss, 
            list_tasks= self.tasks, 
            evaluate_initial_skillFactor= evaluate_initial_skillFactor
        )

        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks) 
        EVAL_PER_EPOCH_PER_TASKS= 100
        eval_each_task = np.zeros(len(self.tasks))

        generation = 0 
        stop = False 
        epoch = 0 

        self.history_cost.append([ind.fcost for ind in population.get_solves()])

        while np.sum(eval_each_task) <= MAXEVALS and stop is False: 
            generation += 1 

            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks), 
                dim = self.dim_uss, 
                list_tasks= self.tasks
            )


            ls_idx_ind = np.arange(np.sum(nb_inds_tasks))
            np.random.shuffle(ls_idx_ind) 

            for idx_ind_task in ls_idx_ind: 
                idx_ind = idx_ind_task % nb_inds_tasks[0] 
                skf_pa = idx_ind_task // nb_inds_tasks[0]
                 

                if sum(eval_each_task) >= epoch * EVAL_PER_EPOCH_PER_TASKS * len(self.tasks):
                    # save history 
                    self.history_cost.append([ind.fcost for ind in population.get_solves()])
                    self.history_rmp.append(self.rmp.get_rmp()) 

                    epoch += 1 
                    self.render_process(
                        curr_progress= epoch / nb_generations, 
                        list_desc= ['Pop_size', 'Cost'], 
                        list_value=[[len(population)], self.history_cost[-1]], 
                        use_sys= True,
                    )

                # 
                skf_pb = numba_randomchoice(len(self.tasks))
                
                rmp = None
                if skf_pb == self.rmp.best_partner[skf_pa]: 
                    rmp = 1 
                else: 
                    mu_rmp = self.rmp.rmp[skf_pa][skf_pb] 

                    rmp = numba_random_gauss(mean= mu_rmp, sigma= 0.1)
                    while rmp <= 0 or rmp > 1: 
                        rmp = numba_random_gauss(mean= mu_rmp, sigma= 0.1)

                random_number = numba_random_uniform()[0]


                pa = population[skf_pa][idx_ind]


                # TM-FIXME: not have mutation 
                if skf_pb == skf_pa or random_number > rmp: 
                    # intra
                    # TM-FIXME: Review for if random_number > rmp should push to best_partner or not ? 
                    pb = population[skf_pa].__getRandomItems__() 

                    if np.all(pa.genes == pb.genes): 
                        pb = population[skf_pa].__getWorstIndividual__ 
                        if np.all(pa.genes == pb.genes):
                            pb = population[skf_pa].getSolveInd() 

                    oa = self.search(pa, population) 

                    if oa is None: 
                        oa, _ = self.crossover(pa, pb, skf_pa, skf_pa, population) 

                    _, ob = self.crossover(pa, pb, skf_pa, skf_pa, population) 

                    offsprings.__addIndividual__(oa) 
                    offsprings.__addIndividual__(ob) 

                else: 
                    pb = population[skf_pb].__getRandomItems__(size=1)[0]

                    oa, ob = self.crossover(pa, pb, skf_pa, skf_pa)

                    offsprings.__addIndividual__(oa) 
                    offsprings.__addIndividual__(ob) 
                    
                    Delta1 = pa.fcost - oa.fcost 
                    Delta2 = pa.fcost - ob.fcost 
                    
                    if Delta1 > 0 or Delta2 > 0: 
                        self.rmp.s_rmp[skf_pa][skf_pb].append(rmp) 
                        self.rmp.diff_f_inter_x[skf_pa][skf_pb].append(max([Delta1, Delta2]))
                    
                
                replace_ind = oa if oa.fcost < ob.fcost else ob

                if replace_ind.fcost < population[skf_pa][idx_ind].fcost: 
                    population[skf_pa][idx_ind].fcost = replace_ind.fcost 
                    population[skf_pa][idx_ind].genes = replace_ind.genes 
                else: 
                    offsprings.__addIndividual__(population[skf_pa].__copyIndividual__(pa))

                eval_each_task[skf_pa] += 2 


            # merge 
            population = offsprings 
            population.update_rank() 

            # TM-TODO: Local search 


            # Selection 

            nb_inds_tasks = [
                int(
                    int(
                        min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* (epoch - 1) + nb_inds_each_task, nb_inds_each_task)
                    )
                )
            ] * len(self.tasks)

            self.selection(population= population, nb_inds_tasks= nb_inds_tasks)

            # update operators 
            self.crossover.update(population) 
            self.mutation.update(population) 
            self.search.update(population) 

            # update rmp 
            self.rmp.update() 


        self.last_pop = population
        self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
        print()
        print(eval_each_task)
        print('END!')
        return self.last_pop.get_solves()



                 
                     
                    
                    



    
