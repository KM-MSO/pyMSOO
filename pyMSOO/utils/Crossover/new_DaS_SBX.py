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

    def __init__(self, nc= 2, eta= 3.0, conf_thres=1):
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
                kl = np.log((std[i] + 1e-7)/(std[j] + 1e-7)) + (std[j] **2 + (mean[j] - mean[i]) ** 2+ 1e-7)/(2 * std[i] ** 2 + 1e-7) - 1/2
                                                                #   2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                                                                  
                # kl = np.log((std[i] + 1e-50)/(std[j] + 1e-50)) + (std[j] ** 2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                prob[i][j] = np.exp(-kl * u)

        return np.clip(prob, 1/(dim_uss*2), 1)

    def update(self, population: Population, **kwargs) -> None:
        mean = np.zeros((self.nb_tasks, self.dim_uss))
        std = np.zeros((self.nb_tasks, self.dim_uss))
        for idx_subPop in range(self.nb_tasks):
            mean[idx_subPop] = population[idx_subPop].__meanInds__
            std[idx_subPop] = population[idx_subPop].__stdInds__
            # std[idx_subPop] = np.where(std[idx_subPop] < 1e-3, 1e-3, std[idx_subPop])
        self.prob = self.__class__._updateProb(
            self.prob, 10**(-self.eta), self.dim_uss, self.nb_tasks, mean, std)

    # @staticmethod
    # @jit(nopython=True)
    # def _crossover(gene_pa, gene_pb, swap, dim_uss, nc):
    #     u = np.random.rand(dim_uss)
    #     beta = np.where(u < 0.5, (2*u)**(1/(nc + 1)),
    #                     (2 * (1 - u))**(-1 / (nc + 1)))

    #     # like pa
    #     gene_oa = np.clip(
    #         0.5*((1 + beta) * gene_pa + (1 - beta) * gene_pb), 0, 1)
    #     # like pb
    #     gene_ob = np.clip(
    #         0.5*((1 - beta) * gene_pa + (1 + beta) * gene_pb), 0, 1)

    #     # swap
    #     if swap:
    #         idx_swap = np.where(np.random.rand(dim_uss) < 0.5)[0]
    #         gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]

    #     return gene_oa, gene_ob

    def choose_task_transfer(self, pcd_vector_skfpa: np.array, transfered_dims: list,  transfered_task: list = [], smp = None,) -> int:
        '''
        pcd_vector_skfpa : shape (K, D) 
        base only on pcd vector 
        Return id task transferred knowledge 

        '''
        pcd_vector_skfpa[:, transfered_dims] = 0  # shape: (K, D)
        pcd_vector_skfpa = np.sum(pcd_vector_skfpa, axis=1)
        pcd_vector_skfpa[transfered_task] = 0
        if smp is not None: 
            pcd_vector_skfpa *= smp[:len(pcd_vector_skfpa)]
        pcd_vector_skfpa = pcd_vector_skfpa ** 2 
        pcd_vector_skfpa = pcd_vector_skfpa / np.sum(pcd_vector_skfpa)
        return numba_randomchoice_w_prob(pcd_vector_skfpa)


        # smp = smp[:len(pcd_vector_skfpa)]
        # smp[transfered_task] = 0 
        # smp /= np.sum(smp)
        # return numba_randomchoice_w_prob(smp)
        # return numba_randomchoice_w_prob(smp_vector)
    

    @staticmethod
    @jit(nopython=True)
    def _crossover(gene_pa, gene_pb, conf_thres, dim_uss, nc, pcd, gene_p_of_oa, gene_p_of_ob, 
                #    transfered_dims, 
                   thresh_pcd_transfer=0, must_transfer=True, swap = False):
        '''
        Return gene_oa, gene_ob, idx_crossover 
        '''
        u = np.random.rand(dim_uss)
        beta = np.where(u < 0.5, (2*u)**(1/(nc + 1)),
                        (2 * (1 - u))**(-1 / (nc + 1)))

        idx_crossover = np.random.rand(dim_uss) < np.where(
            pcd > thresh_pcd_transfer, pcd, 0)
        # if transfered_dims is not None:
        #     idx_crossover[transfered_dims] = 0
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

        if swap:
            idx_swap = np.where(np.random.rand(dim_uss) < 0.5)[0]
            gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]
    

        return gene_oa, gene_ob, idx_crossover

    def __call__(self, pa: Individual, skf_pb, population: Population, smp: np.array, *args, **kwargs) -> Tuple[Individual, Individual]:
        '''
        '''

        transfered_dims, transferred_tasks = [], [skf_pb, pa.skill_factor]

        pb = population[skf_pb].__getRandomItems__()
        if np.all(pa.genes == pb.genes):
            pb = population[skf_pb].__getWorstIndividual__

        gene_oa, gene_ob, idx_crossover = self.__class__._crossover(pa.genes, pb.genes, 1, self.dim_uss, nc=self.nc,
                                                                    pcd=self.prob[pa.skill_factor][pb.skill_factor], 
                                                                    gene_p_of_oa=pa.genes,
                                                                    gene_p_of_ob=pa.genes,
                                                                    swap= pa.skill_factor == skf_pb, 
                                                                    must_transfer= True,
                                                                    #    transfered_dims= transfered_dims if len(transfered_dims) > 0 else None
                                                                    )
        idx_crossover = np.where(idx_crossover)[0]
        transfered_dims += idx_crossover.tolist()
        count_crossover = 1 if len(transfered_dims) > 0 else 0 

        while count_crossover < self.number_parent:
            if len(transfered_dims) == self.dim_uss:
                break
            if len(transferred_tasks) == len(self.tasks):
                break 
            skf_pc = self.choose_task_transfer(
                pcd_vector_skfpa=self.prob[pa.skill_factor].copy(),
                transfered_dims=transfered_dims,
                transfered_task=transferred_tasks,
                smp= smp
            )
            pc = population[skf_pc].__getRandomItems__()

            gene_oc1, gene_oc2, idx_crossover = self.__class__._crossover(
                gene_pa=pa.genes,
                gene_pb=pc.genes,
                conf_thres=1,
                dim_uss=self.dim_uss,
                nc=self.nc,
                pcd=self.prob[pa.skill_factor][skf_pc],
                gene_p_of_oa=pa.genes,
                gene_p_of_ob=pa.genes,
                # transfered_dims= transfered_dims if len(transfered_dims) > 0 else None,
                thresh_pcd_transfer=0.5,
                must_transfer=False
            )
            idx_crossover[transfered_dims] = 0 
            idx_crossover = np.where(idx_crossover)[0]
            gene_oa[idx_crossover] = gene_oc1[idx_crossover]
            gene_ob[idx_crossover] = gene_oc2[idx_crossover]

 
            count_crossover += 1 

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
