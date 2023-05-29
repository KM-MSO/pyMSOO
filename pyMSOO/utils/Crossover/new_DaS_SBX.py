import numpy as np
from typing import Tuple, Type, List

from ..EA import AbstractTask, Individual, Population
from numba import jit
from .utils import AbstractCrossover

class new_DaS_SBX_Crossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 2, eta = 3, conf_thres= 1):
        self.nc = nc
        self.eta = eta
        self.conf_thres = conf_thres
    
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        # self.prob = 1 - KL_divergence
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, self.dim_uss))

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

    def __das(ind_ab, ind_ac, original_ind, dim_uss, pcd_ab, pcd_ac, idx_transfer_ab):
        if pcd_ac is not None: 
            idx_transfer_ac = np.random.rand(dim_uss) < np.where(pcd_ac > 0.0, pcd_ac, 0)
        else: 
            idx_transfer_ac = np.zeros(dim_uss, dtype= np.int64)
        priority_ab = np.zeros(dim_uss, dtype= np.int64) + 1 

        idx_transfer_ab = np.where(idx_transfer_ab == idx_transfer_ac, priority_ab * idx_transfer_ab, idx_transfer_ab)
        idx_transfer_ac = np.where(idx_transfer_ab == idx_transfer_ac, (1 - priority_ab) * idx_transfer_ac, idx_transfer_ac)



        if np.all(idx_transfer_ab == 0) or np.all(ind_ab[idx_transfer_ab] == original_ind[idx_transfer_ab]):
            # alway crossover -> new individual
            idx_notsame = np.where(ind_ab != original_ind)[0]
            if len(idx_notsame) == 0:
                idx_transfer_ab = np.ones((dim_uss, ), dtype= np.bool_)
            else:
                idx_transfer_ab[np.random.choice(idx_notsame)] = True
            
        # if np.all(idx_transfer_ac == 0) or np.all(ind_ac[idx_transfer_ac] == original_ind[idx_transfer_ac]):
        #     # alway crossover -> new individual
        #     idx_notsame = np.where(ind_ac != original_ind)[0]
        #     if len(idx_notsame) == 0:
        #         idx_transfer_ac = np.ones((dim_uss, ), dtype= np.bool_)
        #     else:
        #         idx_transfer_ac[np.random.choice(idx_notsame)] = True
        
        new_ind_genes = np.where(idx_transfer_ab, ind_ab, original_ind)
        if pcd_ac is not None: 
            new_ind_genes = np.where(idx_transfer_ac, ind_ac, new_ind_genes)
        
        return new_ind_genes, np.sum(idx_transfer_ab), np.sum(idx_transfer_ac) 
        
    def __call__(self, pa: Individual, skf_pb, population: Population, PCD_pa,  *args, **kwargs) -> Tuple[Individual, Individual]:
        '''
        choose skf_pc base on skf_pb 
        get random two individual in skf_pb, skf_pc => pb, pc 

        sbx_crossover (pa, pb) => o1_ab, o2_ab 
        sbx_crossover (pa, pc) => o1_ac, o2_ac 
        take only dimension allow transfer in 

        '''
        pcd_pa_pb = PCD_pa[skf_pb].copy()
        pcd_pa = PCD_pa.copy()
        
         
        # take skf_pc 
        ## compute mse_los -> take loss max 
        idx_transfer_ab = np.random.rand(len(pa)) < pcd_pa_pb  # [1, 0, 0, 1]

        if np.sum((1 - idx_transfer_ab)) > 0: # ton tai chieu khong transfer
            pcd_pa[pa.skill_factor] = 0  
            mse_loss = np.sum((pcd_pa * (1 - idx_transfer_ab)), axis= 1) / np.sum((1 - idx_transfer_ab))
            skf_pc = np.argmax(mse_loss) 
        else: 
            skf_pc = None 
        # if np.all(thresshold_skf_pc < 0.5): 
        #     skf_pc = None
        if skf_pb == pa.skill_factor or skf_pc == pa.skill_factor: 
            skf_pc = None 
        
        # if skf_pc is not None and mse_loss[skf_pc] < 0.5: 
        #     skf_pc = None 
        
        pb = population[skf_pb].__getRandomItems__() 

        genes_o1_ab, genes_o2_ab = self.__class__._crossover(pa.genes, pb.genes, pa.skill_factor == pb.skill_factor, 50, 2)
        if skf_pc is not None: 
            pc = population[skf_pc].__getRandomItems__() 
            genes_o1_ac, genes_o2_ac = self.__class__._crossover(pa.genes, pc.genes, pa.skill_factor == pc.skill_factor, 50, 2) 
        else: 
            genes_o1_ac, genes_o2_ac = None, None 
        
        if skf_pc is not None: 
            gene_oa, len_ab, len_ac = self.__class__.__das(genes_o1_ab, genes_o1_ac, pa.genes, 50, PCD_pa[skf_pb], PCD_pa[skf_pc], idx_transfer_ab)
            gene_ob, len_ab2, len_ac2 = self.__class__.__das(genes_o2_ab, genes_o2_ac, pa.genes, 50, PCD_pa[skf_pb], PCD_pa[skf_pc], idx_transfer_ab)
        else:
            gene_oa, len_ab, len_ac = self.__class__.__das(genes_o1_ab, genes_o1_ac, pa.genes, 50, PCD_pa[skf_pb], None, idx_transfer_ab)
            gene_ob, len_ab2, len_ac2 = self.__class__.__das(genes_o2_ab, genes_o2_ac, pa.genes, 50, PCD_pa[skf_pb], None, idx_transfer_ab)

        oa = self.IndClass(gene_oa)
        ob = self.IndClass(gene_ob)

        oa.skill_factor = pa.skill_factor
        ob.skill_factor = pa.skill_factor

        return oa, ob, skf_pc
