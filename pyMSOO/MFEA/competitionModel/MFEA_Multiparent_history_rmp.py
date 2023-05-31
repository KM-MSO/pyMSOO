import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import random

from ...utils.EA import *
from ...utils import Crossover, Mutation, DimensionAwareStrategy, Selection
from ...utils.Search.DifferentialEvolution.shade import *
from ..model import AbstractModel
from ...utils.numba_utils import numba_randomchoice_w_prob


class model(AbstractModel.model):
    # class intra_inter_matrix:
        # def __init__(self, lr, mu= 0.1) -> None:
        #     self.lr = lr
        #     self.mu = mu
        #     self.crossover_rate = 0.5

        # def update_crossover_rate(self, delta_task, count_delta_tasks):
        #     '''
        #     '''

        #     if np.sum(delta_task) != 0:
        #         '''
        #         delta_task : [intra, inter]
        #         '''
        #         new_crossover_rate = np.array(delta_task) / (np.array(count_delta_tasks) + 1e-50)
        #         new_crossover_rate = new_crossover_rate / (np.sum(new_crossover_rate) + 1e-50)
        #         new_crossover_rate = new_crossover_rate[0] # take only intra
        #         self.crossover_rate = self.crossover_rate * (1 - self.lr) + new_crossover_rate * self.lr
        #         if self.crossover_rate < self.mu:
        #             self.crossover_rate = self.mu
        #         if self.crossover_rate > 1 - self.mu:
        #             self.crossover_rate = 1 - self.mu
        #     else:
        #         self.crossover_rate = self.crossover_rate * ( 1- self.lr) + 0.5 * self.lr

        # def get_crossover_rate(self):
        #     return self.crossover_rate

    class intra_inter_matrix:
        '''
        Follow history memory
        '''

        def __init__(self, len_history_mem=30, sigma_random=0.1) -> None:
            self.len_history_mem = len_history_mem
            self.sigma_random = sigma_random
            self.index_update = 0
            self.history_mem = np.zeros(
                (self.len_history_mem), dtype=float) + 0.5

            pass

        def get_crossover_rate(self):

            cr_rate = np.random.normal(loc=self.history_mem[np.random.choice(
                np.arange(self.len_history_mem))], scale=self.sigma_random)
            if cr_rate > 1:
                return 1
            if cr_rate < 0:
                return 0
            return cr_rate

        def update_value_history_mem(self, value):
            self.history_mem[self.index_update] = value
            self.index_update = (self.index_update +
                                    1) % self.len_history_mem

        def udpate(self, Delta, ls_success_crossover_rate):
            'cal new _crossover_rate to push history memory'
            if len(Delta) != 0:
                new_value = np.sum(np.array(Delta) * np.array(ls_success_crossover_rate) ** 2) / np.sum(
                    np.array(Delta) * (np.array(ls_success_crossover_rate)) + 1e-10)
                if new_value < 0.1: 
                    new_value = 0.1 
                if new_value > 0.9: 
                    new_value = 0.9 
            new_value = 0.5
            self.update_value_history_mem(new_value)
            
            
    def __init__(self, seed=None, percent_print=1) -> None:
        super().__init__(seed, percent_print)
        self.ls_attr_avg.append('history_smp')

    def compile(self,
                IndClass: Type[Individual],
                tasks: List[AbstractTask],
                crossover: Crossover.SBX_Crossover,
                mutation: Mutation.PolynomialMutation,
                multi_parent: Crossover.MultiparentCrossover,
                dimension_strategy: DimensionAwareStrategy.AbstractDaS = DimensionAwareStrategy.NoDaS(),
                selection: Selection.AbstractSelection = Selection.ElitismSelection(),
                *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation,
                        dimension_strategy, selection, *args, **kwargs)

        self.multi_parent_crossover = multi_parent
        self.multi_parent_crossover.getInforTasks(IndClass, tasks, self.seed)

    def render_smp(self,  shape=None, title=None, figsize=None, dpi=100, step=1, re_fig=False, label_shape=None, label_loc=None):

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
            figsize = (shape[1] * 6, shape[0] * 5)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle(title, size=15)
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        his_smp: np.ndarray = np.copy(self.history_smp)
        y_lim = (-0.1, 1.1)

        for idx_task, task in enumerate(self.tasks):
            fig.axes[idx_task].stackplot(
                np.append(np.arange(0, len(his_smp), step),
                          np.array([len(his_smp) - 1])),
                [his_smp[
                    np.append(np.arange(0, len(his_smp), step),
                              np.array([len(his_smp) - 1])),
                    idx_task, t] for t in range(len(self.tasks) + 1)],
                labels=['Task' + str(i + 1)
                        for i in range(len(self.tasks))] + ["mutation"]
            )
            # plt.legend()
            fig.axes[idx_task].set_title(
                'Task ' + str(idx_task + 1) + ": " + task.name)
            fig.axes[idx_task].set_xlabel('Generations')
            fig.axes[idx_task].set_ylabel("SMP")
            fig.axes[idx_task].set_ylim(bottom=y_lim[0], top=y_lim[1])

        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(lines, labels, loc=label_loc, ncol=label_shape[1])
        plt.show()
        if re_fig:
            return fig

    def fit(self, nb_generations: int, nb_inds_each_task: int, nb_inds_min=None,
            lr=0.1, mu=0.1,
            evaluate_initial_skillFactor=False,
            *args, **kwargs):
        super().fit(*args, **kwargs)

        # nb_inds_min
        if nb_inds_min is not None:
            assert nb_inds_each_task >= nb_inds_min
        else:
            nb_inds_min = nb_inds_each_task

        # initial history of smp -> for render
        self.history_smp = []

        # initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks=[nb_inds_each_task] * len(self.tasks),
            dim=self.dim_uss,
            list_tasks=self.tasks,
            evaluate_initial_skillFactor=evaluate_initial_skillFactor
        )

        nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)

        # SA params:
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        eval_k = [0] * len(self.tasks)
        epoch = 0

        # prob choose first parent
        p_choose_father = np.ones((len(self.tasks), )) / len(self.tasks)

        # Initialize memory M_smp
        M_smp = [self.intra_inter_matrix(lr, mu)
                 for i in range(len(self.tasks))]

        # save history
        self.history_cost.append(
            [ind.fcost for ind in population.get_solves()])
        self.history_smp.append([M_smp[i].get_crossover_rate()
                                for i in range(len(self.tasks))])

        epoch = 1
        average_rate_success = []
        ls_count_success = []
        ls_count_Delta = []

        while sum(eval_k) <= MAXEVALS:
            turn_eval = 0

            # Delta epoch
            Delta: List[List[float]] = np.zeros((len(self.tasks), 2)).tolist()
            count_Delta: List[List[float]] = np.zeros(
                (len(self.tasks), 2)).tolist()
            count_success = np.zeros((len(self.tasks), 2)).tolist()
            record_success_cr_rate = np.empty((len(self.tasks), 0)).tolist()
            delta_for_cr_rate = np.empty((len(self.tasks), 0)).tolist()
            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks=[0] * len(self.tasks),
                dim=self.dim_uss,
                list_tasks=self.tasks,
            )

            while turn_eval < sum(nb_inds_tasks):
                self.history_smp.append(
                    [M_smp[i].get_crossover_rate() for i in range(len(self.tasks))])
                if sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history
                    self.history_cost.append(
                        [ind.fcost for ind in population.get_solves()])

                    self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [
                                        [len(population)], self.history_cost[-1]], use_sys=True)
                    epoch += 1

                # choose subpop of father pa
                skf_pa = numba_randomchoice_w_prob(p_choose_father)

                inter = False
                oa, ob = None, None
                pa = population[skf_pa].__getRandomItems__()
                cr_rate =  M_smp[skf_pa].get_crossover_rate() 
                if np.random.rand() < cr_rate:
                    # TM-TODO intra
                    pa, pb = population[skf_pa].__getRandomItems__(
                        size=2, replace=False)

                    oa, ob = self.crossover(pa, pb, skf_pa, skf_pa, population)
                    pass
                else:  # inter
                    # TM-TODO
                    oa, ob = self.multi_parent_crossover(pa, population)
                    inter = True
                    pass
                inter = int(inter)
                # add oa, ob to offsprings population and eval fcost
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)

                count_Delta[skf_pa][int(inter)] += 2
                eval_k[skf_pa] += 2
                turn_eval += 2

                # Calculate the maximum improvement percetage
                Delta1 = (pa.fcost - oa.fcost) / (pa.fcost ** 2 + 1e-50)
                Delta2 = (pa.fcost - ob.fcost) / (pa.fcost ** 2 + 1e-50)

                Delta[skf_pa][inter] += max([Delta1, 0])**2
                Delta[skf_pa][inter] += max([Delta2, 0])**2

                if Delta1 > 0:
                    count_success[skf_pa][inter] += 1
                if Delta2 > 0:
                    count_success[skf_pa][inter] += 1
                if Delta1 > 0 or Delta2 > 0: 
                    delta_for_cr_rate[skf_pa].append(max([Delta1, Delta2, 0]))
                    record_success_cr_rate[skf_pa].append(cr_rate) 
            
                    

            average_rate_success.append(
                np.array(count_success) / (np.array(count_Delta) + 0.00001))
            ls_count_success.append(count_success)
            ls_count_Delta.append(count_Delta)

            # merge
            population = population + offsprings
            population.update_rank()

            # selection
            nb_inds_tasks = [int(
                int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)
                    * (epoch - 1) + nb_inds_each_task, nb_inds_each_task))
            )] * len(self.tasks)
            self.selection(population, nb_inds_tasks)

            # update operators
            self.crossover.update(population=population)
            self.mutation.update(population=population)
            self.dimension_strategy.update(population=population)
            self.multi_parent_crossover.update(population=population)



            # update smp
            for skf in range(len(self.tasks)):
                # M_smp[skf].update_crossover_rate(Delta[skf], count_Delta[skf])
                M_smp[skf].udpate(delta_for_cr_rate[skf], record_success_cr_rate[skf])

        np.save("avarage_success.npy", average_rate_success)
        np.save("ls_count_success.npy", ls_count_success)
        np.save("ls_count_Delta.npy", ls_count_Delta)
