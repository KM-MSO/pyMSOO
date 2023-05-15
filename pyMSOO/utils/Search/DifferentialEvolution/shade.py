from pyMSOO.utils.EA import Individual
from ..Abstract import AbstractSearch
from ...EA import AbstractTask, Individual, Population

from typing import Type, List
import numpy as np
import scipy.stats
from ...numba_utils import * 
import time

class NoSearch(AbstractSearch): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def __call__(self, ind, *args, **kwargs) -> Individual:
        return ind 

    

class SHADE(AbstractSearch):
    def __init__(self, len_mem = 30, p_best_type:str = 'ontop', p_ontop = 0.1, tournament_size = 2) -> None:
        '''
        `p_best_type`: `random` || `tournament` || `ontop`
        '''
        super().__init__()
        self.len_mem = len_mem
        self.p_best_type = p_best_type
        self.p_ontop = p_ontop
        self.tournament_size = tournament_size

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        # memory of cr and F in epoch
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        for skf in range(self.nb_tasks):
            new_cr = self.M_cr[skf][self.index_update[skf]]
            new_F = self.M_F[skf][self.index_update[skf]]

            new_index = (self.index_update[skf] + 1) % self.len_mem

            if len(self.epoch_M_cr) > 0:
                new_cr = np.sum(np.array(self.epoch_M_cr[skf]) * (np.array(self.epoch_M_w[skf]) / (np.sum(self.epoch_M_w[skf]) + 1e-50)))
                new_F = np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf]) ** 2) / \
                    (np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf])) + 1e-50)
            
            self.M_cr[skf][new_index] = new_cr
            self.M_F[skf][new_index] = new_F

            self.index_update[skf] = new_index

        # reset epoch mem
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        
    def __call__(self, ind: Individual, population: Population, *args, **kwargs) -> Individual:
        super().__call__(*args, **kwargs)
        # random individual
        ind_ran1, ind_ran2 = population.__getIndsTask__(ind.skill_factor, size = 2, replace= False, type= 'random')


        if np.all(ind_ran1.genes == ind_ran2.genes):
            ind_ran2 = population[ind.skill_factor].__getWorstIndividual__

        # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)
        while ind_best is ind:
            ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)


        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0 or F > 1:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

        new_genes = np.where(u, 
            ind_best.genes + F * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        new_genes = np.clip(new_genes, 0, 1)

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory
        delta = ind.fcost - new_ind.fcost
        if delta > 0:
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

        return new_ind

class L_SHADE(AbstractSearch):
    def __init__(self, len_mem = 30, p_ontop = 0.1, tournament_size = 2) -> None:
        '''
        `p_best_type`: `random` || `tournament` || `ontop`
        '''
        super().__init__()
        self.len_mem = len_mem
        self.p_ontop = p_ontop
        self.tournament_size = tournament_size

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        # memory of cr and F in epoch
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        for skf in range(self.nb_tasks):
            new_cr = self.M_cr[skf][self.index_update[skf]]
            new_F = self.M_F[skf][self.index_update[skf]]

            new_index = (self.index_update[skf] + 1) % self.len_mem

            if len(self.epoch_M_cr) > 0:
                new_cr = np.sum(np.array(self.epoch_M_cr[skf]) * (np.array(self.epoch_M_w[skf]) / (np.sum(self.epoch_M_w[skf]) + 1e-50)))
                new_F = np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf]) ** 2) / \
                    (np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf])) + 1e-50)
            
            self.M_cr[skf][new_index] = new_cr
            self.M_F[skf][new_index] = new_F

            self.index_update[skf] = new_index

        # reset epoch mem
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        
    def __call__(self, ind: Individual, population: Population, *args, **kwargs) -> Individual:
        super().__call__(*args, **kwargs)
        # random individual
        ind_ran1, ind_ran2 = population.__getIndsTask__(ind.skill_factor, size = 2, replace= False, type= 'random')


        if np.all(ind_ran1.genes == ind_ran2.genes):
            ind_ran2 = population[ind.skill_factor].__getWorstIndividual__

        # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, type = 'ontop', p_ontop= self.p_ontop)

        if ind_best is ind:
            if len(population[ind.skill_factor]) * self.p_ontop < 2:
                while ind_best is ind:
                    ind_best = population.__getIndsTask__(ind.skill_factor, type = 'tournament', tournament_size= self.tournament_size)

            else:
                while ind_best is ind:
                    ind_best = population.__getIndsTask__(ind.skill_factor, type = 'ontop', p_ontop= self.p_ontop)


        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0 or F > 1:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

        new_genes = np.where(u, 
            ind.genes + F * (ind_best.genes - ind.genes + ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )

        # u = np.random.rand(self.dim_uss)
        # tmp = ind.genes * u
        # new_genes = np.where(new_genes > 1,tmp + 1 - u, new_genes) 
        # new_genes = np.where(new_genes < 0, tmp, new_genes) 

        new_genes = np.where(new_genes > 1, (ind.genes + 1)/2, new_genes) 
        new_genes = np.where(new_genes < 0, ind.genes / 2, new_genes) 

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory
        delta = ind.fcost - new_ind.fcost
        if delta > 0:
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

        return new_ind


@jit(nopython=True)
def produce_inds(ind_genes, best_genes, ind1_genes, ind2_genes, F, u):
    new_genes = np.where(u,
        ind_genes + F * (best_genes - ind_genes + ind1_genes - ind2_genes),
        ind_genes
    )
    new_genes = np.where(new_genes > 1, (ind_genes + 1)/2, new_genes) 
    new_genes = np.where(new_genes < 0, (ind_genes + 0)/2, new_genes)

    return new_genes

class LSHADE_LSA21(AbstractSearch): 
    def __init__(self, len_mem = 30, p_ontop = 0.1) -> None:
        super().__init__()
        self.len_mem = len_mem 
        self.p_ontop = p_ontop 
        self.archive: list[list[Individual]] = None 
        self.arc_rate = 5 

        self.first_run = True 



    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)

        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        self.archive = np.empty(shape= (self.nb_tasks, 0)).tolist() 

        # memory of cr and F in epoch
        self.epoch_M_cr:list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def __call__(self,ind: Individual, population: Population, *args, **kwargs) -> Individual: 
        super().__call__(*args, **kwargs)

        k = numba_randomchoice(self.len_mem)
        # cr = numba_clip()

        cr = numba_random_gauss(mean = self.M_cr[ind.skill_factor][k], sigma= 0.1)
        if cr < 0: cr = 0 
        elif cr > 1: cr = 1
        # cr = np.clip(numba_random_gauss(mean = self.M_cr[ind.skill_factor][k], sigma= 0.1), 0, 1)
        F = 0
        
        while F <= 0:
            
            F= numba_random_cauchy(self.M_F[ind.skill_factor][k], 0.1)
            # F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
            
        
        if F >1: 
            F = 1 
        
        u = (numba_random_uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[numba_randomchoice(self.dim_uss)] = 1
                # get best individual

        ind_best = population.__getIndsTask__(ind.skill_factor, p_ontop= self.p_ontop)
        while ind_best is ind:
            ind_best = population.__getIndsTask__(ind.skill_factor, p_ontop= self.p_ontop)
        
        ind1 = ind_best 
        while ind1 is ind_best or ind1 is ind : 
            ind1 = population.__getIndsTask__(ind.skill_factor, type='random') 
        

        if self.first_run is False and numba_random_uniform()[0] < len(self.archive[ind.skill_factor]) / (len(self.archive[ind.skill_factor]) + len(population[ind.skill_factor])): 
            ind2 = self.archive[ind.skill_factor][numba_randomchoice(len(self.archive[ind.skill_factor]))]
        else: 
            ind2 = ind1 
            while ind2 is ind_best or ind2 is ind1 or ind2 is ind: 
                ind2 = population.__getIndsTask__(ind.skill_factor, type='random') 
        

        new_genes = produce_inds(ind.genes, ind_best.genes, ind1.genes, ind2.genes, F, u)
        # new_genes = np.where(u, 
        #     ind.genes + F * (ind_best.genes - ind.genes + ind1.genes - ind2.genes),
        #     ind.genes
        # )
        # new_genes = np.where(new_genes > 1, (ind.genes + 1)/2, new_genes) 
        # new_genes = np.where(new_genes < 0, (ind.genes + 0)/2, new_genes) 
        if np.all(new_genes == ind.genes): 
            return None 
        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor

        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])
        
        # save memory 
        delta = ind.fcost - new_ind.fcost 
        if delta == 0 : 
            return new_ind 
        elif delta > 0: 
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

            if len(self.archive[ind.skill_factor]) < self.arc_rate * len(population[ind.skill_factor]): 
                self.archive[ind.skill_factor].append(ind)
            else: 
                del self.archive[ind.skill_factor][numba_randomchoice(len(self.archive[ind.skill_factor]))]
                self.archive[ind.skill_factor].append(ind)
            return new_ind 
        return new_ind


    def update(self, population, *args, **kwargs) -> None:
        self.first_run = False 
        for skf in range(self.nb_tasks): 
            if(len(self.epoch_M_cr[skf])) > 0: 
                sum_diff = np.sum(np.array(self.epoch_M_w[skf])) 
                w = np.array(self.epoch_M_w[skf]) / sum_diff 

                tmp_sum_cr = np.sum(w * np.array(self.epoch_M_cr[skf]))
                tmp_sum_f = np.sum(w * np.array(self.epoch_M_F[skf])) 


                self.M_F[skf][self.index_update[skf]] = np.sum(w * np.array(self.epoch_M_F[skf]) ** 2) / tmp_sum_f

                if (tmp_sum_cr == 0): 
                    self.M_cr[skf][self.index_update[skf]] = -1 
                else: 
                    self.M_cr[skf][self.index_update[skf]] = np.sum(w * np.array(self.epoch_M_cr[skf]) ** 2) / tmp_sum_cr 
                
                self.index_update[skf] = (self.index_update[skf] + 1) % self.len_mem
            
        
        # reset epoch mem
        self.epoch_M_cr:list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # update archive size
        for skf in range(self.nb_tasks): 
            while len(self.archive[skf]) > len(population[skf]) * self.arc_rate: 
                del self.archive[skf][numba_randomchoice(len(self.archive[skf]))]
