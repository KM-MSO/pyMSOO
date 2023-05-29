import numpy as np
from typing import Tuple, Type, List

from ..EA import AbstractTask, Individual, Population
from numba import jit
from .utils import AbstractCrossover
from ...utils.numba_utils import numba_randomchoice_w_prob

class MultiparentCrossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 2, eta = 3, conf_thres= 1):
        self.nc = nc
        self.eta = eta
        self.conf_thres = conf_thres
        self.prob = None 
        self.prob_in_dim = None 
    
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        # self.prob = 1 - KL_divergence
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.prob_in_dim = np.ones((self.nb_tasks, self.dim_uss, self.nb_tasks)) # n tac vu, 1 chieu tai 1 tac vu -> wheel prob length of nb_tasks ? except chinh no :)) 

        # TM-TODO: optimize code 
        for idx_task in range(self.nb_tasks):
            for dim in range(self.dim_uss):
                tmp = self.prob[idx_task][idx_task][dim] 
                self.prob[idx_task][idx_task][dim] = 0 
                self.prob_in_dim[idx_task][dim] = self.prob[idx_task, :, dim] / np.sum(self.prob[idx_task, :, dim]) 
                self.prob[idx_task][idx_task][dim] = tmp 
        


    @staticmethod
    @jit(nopython= True)
    def _updateProb(prob, u, dim_uss, nb_tasks, mean, std):
        for i in range(nb_tasks):
            for j in range(nb_tasks):
                kl = np.log((std[i] + 1e-50)/(std[j] + 1e-50)) + (std[j] ** 2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                prob[i][j] = np.exp(-kl * u)

        return np.clip(prob, 1/dim_uss, 1)

    def update(self, population: Population, **kwargs) -> None:
        mean = np.zeros((self.nb_tasks, self.dim_uss))
        std = np.zeros((self.nb_tasks, self.dim_uss))
        for idx_subPop in range(self.nb_tasks):
            mean[idx_subPop] = population[idx_subPop].__meanInds__
            std[idx_subPop] = population[idx_subPop].__stdInds__
        self.prob = self.__class__._updateProb(self.prob, 10**(-self.eta), self.dim_uss, self.nb_tasks, mean, std)

        # TM-TODO: optimize code 
        for idx_task in range(self.nb_tasks):
            for dim in range(self.dim_uss):
                tmp = self.prob[idx_task][idx_task][dim] 
                self.prob[idx_task][idx_task][dim] = 0 
                self.prob_in_dim[idx_task][dim] = self.prob[idx_task, :, dim] / np.sum(self.prob[idx_task, :, dim]) 
                self.prob[idx_task][idx_task][dim] = tmp 


    @staticmethod
    @jit(nopython = True)
    def _crossover(gene_pa, gene_pb, swap, dim_uss, nc):
        u = np.random.rand(dim_uss)
        beta = np.where(u < 0.5, (2*u)**(1/(nc +1)), (2 * (1 - u))**(-1 / (nc + 1)))

        #like pa
        gene_oa = np.clip(0.5*((1 + beta) * gene_pa + (1 - beta) * gene_pb), 0, 1)
        #like pb
        gene_ob = np.clip(0.5*((1 - beta) * gene_pa + (1 + beta) * gene_pb), 0, 1)

        #swap
        if swap:
            idx_swap = np.where(np.random.rand(dim_uss) < 0.5)[0]
            gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]
    
        return gene_oa, gene_ob

        
    def __call__(self, pa: Individual, population: Population) -> Tuple[Individual, Individual]:
        '''
        Cho lai ghep toan bo chieu 
        '''

        ls_id_skf_cross = [numba_randomchoice_w_prob(self.prob_in_dim[pa.skill_factor][dim]) for dim in range(self.dim_uss)] # [1, 2, 4, 3 ...,] size= dim
        # ls_id_skf_cross = [np.argmax(self.prob_in_dim[pa.skill_factor][dim]) for dim in range(self.dim_uss)]
        prob_cross = [] 
        for idx_dim, idx_skf in enumerate(ls_id_skf_cross):
            prob_cross.append(self.prob[pa.skill_factor][idx_skf][idx_dim])

        idx_transfer = np.random.rand(self.dim_uss) < np.array(prob_cross) 
        ls_idx_pb = np.random.choice(np.arange(len(population[pa.skill_factor])), size= self.dim_uss, replace= True) 

        pb_genes = [] 
        for idx_dim, id_skf in enumerate(ls_id_skf_cross):
            pb_genes.append(population[id_skf][ls_idx_pb[id_skf]][idx_dim])
        pb_genes = np.array(pb_genes)

        gene_oa, gene_ob = self._crossover(pa.genes, pb_genes, swap= False, dim_uss= self.dim_uss, nc= self.nc)



        if np.all(idx_transfer == 0) or np.all(pb_genes[idx_transfer] == pa.genes[idx_transfer]):
            # alway crossover -> new individual
            idx_notsame = np.where(pb_genes != pa.genes)[0]
            if len(idx_notsame) == 0:
                idx_transfer = np.ones((self.dim_uss, ), dtype= np.bool_)
            else:
                idx_transfer[np.random.choice(idx_notsame)] = True
            

        gene_oa = np.where(idx_transfer, gene_oa, pa.genes)
        gene_ob = np.where(idx_transfer, gene_ob, pa.genes)

    
        oa = self.IndClass(gene_oa)
        ob = self.IndClass(gene_ob)

        oa.skill_factor = pa.skill_factor
        ob.skill_factor = pa.skill_factor

        return oa, ob 
        

        pass 