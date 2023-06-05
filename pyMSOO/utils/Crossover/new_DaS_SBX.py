import numpy as np
from typing import Tuple, Type, List

from ..EA import AbstractTask, Individual, Population
from numba import jit
from .utils import AbstractCrossover
from pyMSOO.utils.numba_utils import numba_randomchoice_w_prob


class new_DaS_SBX_Crossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''

    def __init__(self, nc=2, eta=1.5, conf_thres=1):
        self.nc = nc
        self.eta = eta
        self.conf_thres = conf_thres
        self.prob = None 
        self.number_parent = 2 

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        # self.prob = 1 - KL_divergence
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, self.dim_uss))

    @staticmethod
    @jit(nopython=True)
    def _updateProb(prob, u, dim_uss, nb_tasks, mean, std):
        for i in range(nb_tasks):
            for j in range(nb_tasks):
                kl = np.log((std[i] + 1e-50)/(std[j] + 1e-50)) + (std[j] **
                                                                  2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                prob[i][j] = np.exp(-kl * u)

        return np.clip(prob, 1/dim_uss, 1)

    def update(self, population: Population, **kwargs) -> None:
        mean = np.zeros((self.nb_tasks, self.dim_uss))
        std = np.zeros((self.nb_tasks, self.dim_uss))
        for idx_subPop in range(self.nb_tasks):
            mean[idx_subPop] = population[idx_subPop].__meanInds__
            std[idx_subPop] = population[idx_subPop].__stdInds__
        self.prob = self.__class__._updateProb(
            self.prob, 10**(-self.eta), self.dim_uss, self.nb_tasks, mean, std)

    @staticmethod
    @jit(nopython=True)
    def _crossover(gene_pa, gene_pb, swap, dim_uss, nc):
        u = np.random.rand(dim_uss)
        beta = np.where(u < 0.5, (2*u)**(1/(nc + 1)),
                        (2 * (1 - u))**(-1 / (nc + 1)))

        # like pa
        gene_oa = np.clip(
            0.5*((1 + beta) * gene_pa + (1 - beta) * gene_pb), 0, 1)
        # like pb
        gene_ob = np.clip(
            0.5*((1 - beta) * gene_pa + (1 + beta) * gene_pb), 0, 1)

        # swap
        if swap:
            idx_swap = np.where(np.random.rand(dim_uss) < 0.5)[0]
            gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]

        return gene_oa, gene_ob

    def choose_task_transfer(self, pcd_vector_skfpa: np.array, transfered_dims: list,  transfered_task: list = []) -> int:
        '''
        pcd_vector_skfpa : shape (K, D) 
        base only on pcd vector 
        Return id task transferred knowledge 

        '''
        pcd_vector_skfpa[:, transfered_dims] = 0  # shape: (K, D)
        pcd_vector_skfpa = np.sum(pcd_vector_skfpa, axis=1)
        pcd_vector_skfpa[transfered_task] = 0
        pcd_vector_skfpa = pcd_vector_skfpa / np.sum(pcd_vector_skfpa)

        return numba_randomchoice_w_prob(pcd_vector_skfpa)

        # return numba_randomchoice_w_prob(smp_vector)

    # @staticmethod
    # @jit(nopython=True)
    def _crossover(gene_pa, gene_pb, conf_thres, dim_uss, nc, pcd, gene_p_of_oa, gene_p_of_ob, transfered_dims, thresh_pcd_transfer= 0, must_transfer= True):
        '''
        Return gene_oa, gene_ob, idx_crossover 
        '''
        u = np.random.rand(dim_uss)
        beta = np.where(u < 0.5, (2*u)**(1/(nc + 1)),
                        (2 * (1 - u))**(-1 / (nc + 1)))

        idx_crossover = np.random.rand(dim_uss) < np.where(pcd > thresh_pcd_transfer, pcd, 0)
        idx_crossover[transfered_dims] = 0 
        if must_transfer:
            if np.all(idx_crossover == 0) or np.all(gene_pa[idx_crossover] == gene_pb[idx_crossover]):
                # alway crossover -> new individual
                idx_notsame = np.where(gene_pa != gene_pb)[0]
                if len(idx_notsame) == 0:
                    idx_crossover = np.ones((dim_uss, ), dtype=np.bool_)
                else:
                    idx_crossover[np.random.choice(idx_notsame)] = True

        # like pa
        gene_oa = np.where(idx_crossover, np.clip(
            0.5*((1 + beta) * gene_pa + (1 - beta) * gene_pb), 0, 1), gene_p_of_oa)
        # like pb
        gene_ob = np.where(idx_crossover, np.clip(
            0.5*((1 - beta) * gene_pa + (1 + beta) * gene_pb), 0, 1), gene_p_of_ob)

        # swap
        idx_swap = np.where(np.logical_and(
            np.random.rand(dim_uss) < 0.5, pcd >= conf_thres))[0]
        gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]

        return gene_oa, gene_ob, np.where(idx_crossover)[0]


    def __call__(self, pa: Individual, skf_pb, population: Population, *args, **kwargs) -> Tuple[Individual, Individual]:
        '''
        '''

        transfered_dims, transferred_tasks = [], [skf_pb]

        pb = population[skf_pb].__getRandomItems__()

        gene_oa, gene_ob, idx_crossover = self.__class__._crossover(pa.genes, pb.genes, 1, self.dim_uss, nc=self.nc,
                                           pcd=self.prob[pa.skill_factor][pb.skill_factor], gene_p_of_oa=pa.genes, gene_p_of_ob=pa.genes, 
                                           transfered_dims= transfered_dims
                                           )
        
        transfered_dims += idx_crossover.tolist() 
        for i in range(self.number_parent - 1): 
            if len(transfered_dims) == self.dim_uss: 
                break 
            skf_pc = self.choose_task_transfer(
                pcd_vector_skfpa= self.prob[pa.skill_factor].copy(), 
                transfered_dims= transfered_dims, 
                transfered_task= transferred_tasks
            )
            pc = population[skf_pc].__getRandomItems__()

            gene_oc1, gene_oc2, idx_crossover = self.__class__._crossover(
                gene_pa= pa.genes, 
                gene_pb = pc.genes, 
                conf_thres= 1, 
                dim_uss= self.dim_uss, 
                nc= self.nc, 
                pcd= self.prob[pa.skill_factor][skf_pc],
                gene_p_of_oa= pa.genes, 
                gene_p_of_ob= pa.genes, 
                transfered_dims= transfered_dims, 
                thresh_pcd_transfer= 0.5,
                must_transfer= False

            )

            gene_oa[idx_crossover] = gene_oc1[idx_crossover]
            gene_ob[idx_crossover] = gene_oc2[idx_crossover]



            # gene_oa, _, idx_crossover = self.__class__._crossover(
            #     gene_pa = gene_oa, 
            #     gene_pb = pc_oa, 
            #     conf_thres= 1, 
            #     dim_uss= self.dim_uss, 
            #     nc= self.nc, 
            #     pcd= self.prob[pa.skill_factor][skf_pc],
            #     gene_p_of_oa= gene_oa, 
            #     gene_p_of_ob= gene_oa, 
            #     transfered_dims= transfered_dims
            # )
            # gene_ob, _, idx_crossover = self.__class__._crossover(
            #     gene_pa = gene_ob, 
            #     gene_pb = pc_ob, 
            #     conf_thres= 1, 
            #     dim_uss= self.dim_uss, 
            #     nc= self.nc, 
            #     pcd= self.prob[pa.skill_factor][skf_pc],
            #     gene_p_of_oa= gene_ob, 
            #     gene_p_of_ob= gene_ob, 
            #     transfered_dims= transfered_dims
            # )

            # gene_oa, gene_ob, idx_crossover = self.__class__._crossover(gene_oa, 
            #                                         gene_pb= gene_ob, 
            #                                         conf_thres= 1, 
            #                                         dim_uss = self.dim_uss, 
            #                                         nc= self.nc, 
            #                                         pcd = self.prob[pa.skill_factor][skf_pc],
            #                                         gene_p_of_oa= gene_oa, 
            #                                         gene_p_of_ob= gene_ob,
            #                                         transfered_dims= transfered_dims
            #                                     )

            transfered_dims += idx_crossover.tolist()
            transferred_tasks.append(skf_pc)

        oa = self.IndClass(gene_oa)
        ob = self.IndClass(gene_ob)

        oa.skill_factor = pa.skill_factor
        ob.skill_factor = pa.skill_factor 

        return oa, ob
