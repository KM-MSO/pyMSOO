import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import random

from ...utils.EA import *
from ...utils import Crossover, Mutation, DimensionAwareStrategy, Selection, Search
from ...utils.Search.DifferentialEvolution.shade import * 
from ..model import AbstractModel
from ...utils.numba_utils import numba_randomchoice_w_prob

class model(AbstractModel.model):
    class battle_smp:
        def __init__(self, idx_host: int, nb_tasks: int, lr, mu=0.1) -> None:
            assert idx_host < nb_tasks
            self.idx_host = idx_host
            self.nb_tasks = nb_tasks
            self.lr = lr
            self.mu = mu

            #value const for intra
            self.lowerBound_smp = mu/(self.nb_tasks + 1)

            # smp without lowerbound
            self.smp_wo_mu: np.ndarray = np.zeros((nb_tasks + 1, )) + (1 - self.mu)/(nb_tasks + 1)
            self.smp_wo_mu[self.idx_host] += (1 - self.mu) - np.sum(self.smp_wo_mu)

            self.smp_vector = self.smp_wo_mu + self.lowerBound_smp
            
        def get_smp(self) -> np.ndarray:
            return np.copy(self.smp_vector)
        
        def update_SMP(self, Delta_task, count_Delta_tasks):
            '''
            Delta_task > 0 
            '''

            if np.sum(Delta_task) != 0:         
                new_smp = np.array(Delta_task) / (np.array(count_Delta_tasks) + 1e-50)
                new_smp = new_smp * (1 - self.mu) / (np.sum(new_smp) + 1e-50)

                self.smp_wo_mu = self.smp_wo_mu * (1 - self.lr) + new_smp * self.lr
                self.smp_wo_mu[self.idx_host] += (1 - self.mu) - np.sum(self.smp_wo_mu)
                
                self.smp_vector = self.smp_wo_mu + self.lowerBound_smp
            return self.smp_vector
    
    def __init__(self, seed=None, percent_print=2) -> None:
        super().__init__(seed, percent_print)
        self.ls_attr_avg.append('history_smp')

    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, 
        mutation: Mutation.PolynomialMutation, 
        search: Search.DifferentialEvolution.LSHADE_LSA21,
        dimension_strategy: DimensionAwareStrategy.AbstractDaS = DimensionAwareStrategy.NoDaS(),
        selection: Selection.AbstractSelection= Selection.ElitismSelection(), 
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, dimension_strategy, selection, *args, **kwargs)
        self.search = search 
        self.search.getInforTasks(IndClass, tasks, seed = self.seed)

    def render_smp(self,  shape = None, title = None, figsize = None, dpi = 100, step = 1, re_fig = False, label_shape= None, label_loc= None):
        
        if title is None:
            title = self.__class__.__name__
        if shape is None:
            shape = (int(np.ceil(len(self.tasks) / 3)), 3)
        else:
            assert shape[0] * shape[1] >= len(self.tasks)

        if label_shape is None:
            label_shape = (1, len(self.tasks))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.tasks)

        if label_loc is None:
            label_loc = 'lower center'

        if figsize is None:
            figsize = (shape[1]* 6, shape[0] * 5)

        fig = plt.figure(figsize= figsize, dpi = dpi)
        fig.suptitle(title, size = 15)
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        his_smp:np.ndarray = np.copy(self.history_smp)
        y_lim = (-0.1, 1.1)

        for idx_task, task in enumerate(self.tasks):
            fig.axes[idx_task].stackplot(
                np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
                [his_smp[
                    np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
                    idx_task, t] for t in range(len(self.tasks) + 1)],
                labels = ['Task' + str(i + 1) for i in range(len(self.tasks))] + ["mutation"]
            )
            # plt.legend()
            fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + task.name)
            fig.axes[idx_task].set_xlabel('Generations')
            fig.axes[idx_task].set_ylabel("SMP")
            fig.axes[idx_task].set_ylim(bottom = y_lim[0], top = y_lim[1])


        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(lines, labels, loc = label_loc, ncol = label_shape[1])
        plt.show()
        if re_fig:
            return fig

    def fit(self, nb_generations: int, \
            nb_inds_each_task: int, nb_inds_min = None,
            lr = 0.1, mu= 0.1,
            evaluate_initial_skillFactor = False,
            local_search = True, 
            *args, **kwargs):
        super().fit(*args, **kwargs)
        
        # nb_inds_min
        if nb_inds_min is not None:
            assert nb_inds_each_task >= nb_inds_min
        else: 
            nb_inds_min = nb_inds_each_task

        # initial history of smp -> for render
        self.history_smp = []

        #initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)
        
        # SA params:
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        eval_k = [0] * len(self.tasks)
        epoch = 0

        '''
        ------
        per params
        ------
        '''
        # prob choose first parent
        p_choose_father = np.ones((len(self.tasks), ))/ len(self.tasks)

        # Initialize memory M_smp
        M_smp = [self.battle_smp(i, len(self.tasks), lr, mu) for i in range(len(self.tasks))]

        #save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])
        epoch = 1

        while sum(eval_k) <= MAXEVALS:
            turn_eval = 0

            # Delta epoch
            Delta:List[List[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()
            count_Delta: List[List[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()

            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )

            ls_idx = np.arange(sum(nb_inds_tasks)) 
            np.random.shuffle(ls_idx) 
            for idx_ind_task in ls_idx: 
                idx_ind = idx_ind_task % nb_inds_tasks[0] 
                idx_task = idx_ind_task // nb_inds_tasks[0]

                if sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history
                    self.history_cost.append([ind.fcost for ind in population.get_solves()])
                    self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])

                    self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
                    epoch += 1
                
                skf_pa = idx_task 

                smp = M_smp[skf_pa].get_smp()

                skf_pb = numba_randomchoice_w_prob(smp)

                if skf_pb != len(self.tasks):
                    if skf_pa == skf_pb:
                        pa = population[skf_pa][idx_ind]

                        # TM-ANCHOR: intra-crossover: replace SBX-Crossover by LSHADE
                        oa = self.search(pa, population)
                        # ob = self.search(pb, population)
                        
                        # oa, ob = self.crossover(pa, pb, skf_pa, skf_pa, population)

                        offsprings.__addIndividual__(oa)
                        count_Delta[skf_pa][skf_pb] += 1
                        eval_k[skf_pa] += 1
                        turn_eval += 1

                        Delta1 = (pa.fcost - oa.fcost) / (pa.fcost ** 2 + 1e-50)
                        Delta[skf_pa][skf_pb] += max([Delta1, 0])**2

                        if oa.fcost < pa.fcost: 
                            population[skf_pa][idx_ind].fcost = oa.fcost 
                            population[skf_pa][idx_ind].genes = oa.genes  


                    else:
                        pa = population[skf_pa][idx_ind]
                        pb = population[skf_pb].__getRandomItems__()

                        if np.all(pa.genes == pb.genes):
                            pb = population[skf_pb].__getWorstIndividual__
                        
                        oa, ob = self.crossover(pa, pb, skf_pa, skf_pa, population)

                        # dimension strategy
                        oa = self.dimension_strategy(oa, pb.skill_factor, pa)
                        ob = self.dimension_strategy(ob, pb.skill_factor, pb if skf_pa == skf_pb else pa)
                        
                        # add oa, ob to offsprings population and eval fcost
                        offsprings.__addIndividual__(oa)
                        offsprings.__addIndividual__(ob)
                        offsprings.__addIndividual__(population[skf_pa].__copyIndividual__(pa))

                        count_Delta[skf_pa][skf_pb] += 2
                        eval_k[skf_pa] += 2
                        turn_eval += 2

                        # Calculate the maximum improvement percetage
                        Delta1 = (pa.fcost - oa.fcost) / (pa.fcost ** 2 + 1e-50)
                        Delta2 = (pa.fcost - ob.fcost) / (pa.fcost ** 2 + 1e-50)

                        Delta[skf_pa][skf_pb] += max([Delta1, 0])**2
                        Delta[skf_pa][skf_pb] += max([Delta2, 0])**2

                        if oa.fcost < pa.fcost: 
                            population[skf_pa][idx_ind].genes = oa.genes 
                            population[skf_pa][idx_ind].fcost = oa.fcost 
                        
                        if ob.fcost < population[skf_pa][idx_ind].fcost: 
                            population[skf_pa][idx_ind].genes = ob.genes 
                            population[skf_pa][idx_ind].fcost = ob.fcost 

                else:
                    pa= population[skf_pa][idx_ind]

                    oa = self.mutation(pa, return_newInd= True)
                    oa.skill_factor = skf_pa

                
                    # add oa, ob to offsprings population and eval fcost
                    offsprings.__addIndividual__(oa)
                    offsprings.__addIndividual__(population[skf_pa].__copyIndividual__(pa))

                    count_Delta[skf_pa][skf_pb] += 1
                    eval_k[skf_pa] += 1
                    turn_eval += 1

                    # Calculate the maximum improvement percetage
                    Delta1 = (pa.fcost - oa.fcost) / (pa.fcost ** 2 + 1e-50)

                    Delta[skf_pa][skf_pb] += max([Delta1, 0])**2

                    if oa.fcost < pa.fcost: 
                        population[skf_pa][idx_ind].genes = oa.genes 
                        population[skf_pa][idx_ind].fcost = oa.fcost 

            # merge
            population = offsprings
            population.update_rank()

            # selection
            nb_inds_tasks = [int(
                int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* (epoch - 1) + nb_inds_each_task, nb_inds_each_task))
            )] * len(self.tasks)
            self.selection(population, nb_inds_tasks)

            # update operators
            self.crossover.update(population = population)
            self.mutation.update(population = population)
            self.dimension_strategy.update(population = population)
            self.search.update(population) 


            # update smp
            for skf in range(len(self.tasks)):
                M_smp[skf].update_SMP(Delta[skf], count_Delta[skf])

        #solve
        self.last_pop = population
        self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
        print()
        print(p_choose_father)
        print(eval_k)
        print('END!')
        return self.last_pop.get_solves()
    