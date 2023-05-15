import numpy as np
import random
from numba import jit

from . import AbstractModel
from ...utils import Crossover, Mutation, Selection, DimensionAwareStrategy
from ...utils.Mutation import GaussMutation, PolynomialMutation
from ...utils.EA import *
from ...utils.numba_utils import numba_randomchoice, numba_random_gauss, numba_random_cauchy, numba_random_uniform
from ...utils.Search import *

class model(AbstractModel.model):
    TOLERANCE = 1e-6
    INF = 1e8
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, 
        mutation: Mutation.PolynomialMutation, 
        selection: Selection.ElitismSelection,
        dimension_strategy: DimensionAwareStrategy.AbstractDaS = DimensionAwareStrategy.NoDaS(),
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation,dimension_strategy, selection, *args, **kwargs)
    
    def fit(self, nb_generations, 
            nb_inds_each_task = 100, 
            nb_inds_max = 100,
            nb_inds_min = 20,
            evaluate_initial_skillFactor = True, 
            c = 0.06,
            *args, 
            **kwargs) -> List[Individual]:
        super().fit(*args, **kwargs)

        # nb_inds_min
        if nb_inds_min is not None:
            assert nb_inds_each_task >= nb_inds_min
        else: 
            nb_inds_min = nb_inds_each_task

        self.rmp = np.full((len(self.tasks), len(self.tasks)), 0.3)
        self.learningPhase = [LearningPhase(self.IndClass, self.tasks, t) for t in self.tasks]
        
        # initialize population
        self.population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        self.nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)

        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        self.max_eval_k = [nb_generations * nb_inds_each_task] * len(self.tasks)
        self.eval_k = [0] * len(self.tasks)
        epoch = 1
        
        D0 = self.calculateD(population = np.array([[ind.genes for ind in sub.ls_inds] for sub in self.population]), 
                            population_fitness = np.array([sub.getFitness() for sub in self.population]),
                            best = np.array([sub.__getBestIndividual__.genes for sub in self.population]),)

        while sum(self.eval_k) < MAXEVALS:
            self.delta = [[[] for _ in range(len(self.tasks))] for _ in range(len(self.tasks))]

            self.s_rmp = [[[] for _ in range(len(self.tasks))] for _ in range(len(self.tasks))]

            self.population.update_rank()
            # print(self.eval_k)
            if sum(self.eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                # save history
                self.history_cost.append([ind.fcost for ind in self.population.get_solves()])
                self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[sum(self.nb_inds_tasks)], self.history_cost[-1]], use_sys= True)
                epoch += 1

            # offsprings = self.reproduction(sum(self.nb_inds_tasks), self.population)

            # self.population = self.population + offsprings
            # start = time.time()
            matingPool = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
                evaluate_initial_skillFactor = False
            )

            for idx in range(len(self.tasks)):
                
                idx_inds = np.argsort([ind.fcost for ind in self.population[idx].ls_inds])
                
                for i in idx_inds[:int(len(self.population[idx])/2)]:
                    matingPool.__addIndividual__(self.population[idx].ls_inds[i])

            offsprings = self.reproduction(len(self.population), matingPool)
        
            # # merge and update rank
            self.population = matingPool + offsprings
            self.population.update_rank()
            # end = time.time()
            # print("E: ", end - start)
            # selection
            self.nb_inds_tasks = [int(
                int(max((nb_inds_min - nb_inds_max) * (sum(self.eval_k)/MAXEVALS) + nb_inds_max, nb_inds_min))
            )] * len(self.tasks)
            self.selection(self.population, self.nb_inds_tasks)

            # update operators
            self.crossover.update(population = self.population)
            self.mutation.update(population = self.population)
            self.dimension_strategy.update(population = self.population)
            # start = time.time()
            self.updateRMP(c)
            # end = time.time()
            # print("G: ", end - start)
            # start = time.time()
            self.phaseTwo(D0)
            # end = time.time()
            # print("G: ", end - start)
        # self.phaseTwo(D0)
        print('\nEND!')

        #solve 
        self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[sum(self.nb_inds_tasks)], self.history_cost[-1]], use_sys= True)
        self.population.update_rank()
        self.last_pop = self.population
        return self.last_pop.get_solves()
    
    def reproduction(self, size: int, mating_pool: Population,) -> Population:
        sub_size = int(size/len(self.tasks))
       
        offsprings = Population(self.IndClass,
                                nb_inds_tasks = [0] * len(self.tasks), 
                                dim = self.dim_uss,
                                list_tasks= self.tasks)
        counter = np.zeros((len(self.tasks)))  

        stopping = False
        while not stopping:
            pa, pb = mating_pool.__getRandomInds__(2)
            ta = pa.skill_factor
            tb = pb.skill_factor

            if counter[ta] >= sub_size and counter[tb] >= sub_size:
                continue

            rmpValue = numba_random_gauss(mean = max(self.rmp[ta][tb], self.rmp[tb][ta]), sigma = 0.1)

            if ta == tb:
                # self.eval_k[ta] += 2

                oa, ob = self.crossover(pa, pb)

                oa.skill_factor = ta
                ob.skill_factor = ta

                if self.eval_k[ta] >= self.max_eval_k[ta]:
                    oa.fcost = model.INF
                else:
                    self.eval_k[ta] += 1

                offsprings.__addIndividual__(oa)

                if self.eval_k[tb] >= self.max_eval_k[tb]:
                    ob.fcost = model.INF
                else:
                    self.eval_k[tb] += 1

                offsprings.__addIndividual__(ob)

                counter[ta] += 2

            elif random.random() <= rmpValue:
                off = self.crossover(pa, pb)

                for o in off:
                    if counter[ta] < sub_size and random.random() < self.rmp[ta][tb]/(self.rmp[ta][tb] + self.rmp[tb][ta]):
                        o.skill_factor = ta
                        o = self.dimension_strategy(o, tb, pa)
                        if self.eval_k[ta] >= self.max_eval_k[ta]:
                            o.fcost = model.INF
                        else:
                            self.eval_k[ta] += 1
                            o.fcost = self.tasks[ta](o)

                        offsprings.__addIndividual__(o)
                        
                        counter[ta] += 1
                        # self.eval_k[ta] += 1
                        
                        if pa.fcost > o.fcost:
                            self.delta[ta][tb].append(pa.fcost - o.fcost)
                            self.s_rmp[ta][tb].append(rmpValue)
                    
                    elif counter[tb] < sub_size:
                        o.skill_factor = tb
                        o = self.dimension_strategy(o, ta, pb)
        
                        if self.eval_k[tb] >= self.max_eval_k[tb]:
                            o.fcost = model.INF
                        else:
                            self.eval_k[tb] += 1
                            o.fcost = self.tasks[tb](o)

                        offsprings.__addIndividual__(o)
                        
                        counter[tb] += 1
                        # self.eval_k[tb] += 1

                        if pb.fcost > o.fcost:
                            self.delta[tb][ta].append(pb.fcost - o.fcost)
                            self.s_rmp[tb][ta].append(rmpValue)

            else:
                if counter[ta] < sub_size:
                    paa: Individual = self.population[ta].__getRandomItems__()

                    # while np.array_equal(paa.genes, pa.genes):
                    #     paa: Individual = self.population[ta].__getRandomItems__()
                    
                    oa, _ = self.crossover(pa, paa)
                    oa.skill_factor = ta
                    
                    if self.eval_k[ta] >= self.max_eval_k[ta]:
                        oa.fcost = model.INF
                    else:
                        self.eval_k[ta] += 1
                        oa.fcost = self.tasks[ta](oa)

                    offsprings.__addIndividual__(oa)

                    counter[ta] += 1
                    # self.eval_k[ta] += 1

                if counter[tb] < sub_size:
                    pbb: Individual = self.population[tb].__getRandomItems__()

                    # while np.array_equal(pbb.genes, pb.genes):
                    #     pbb: Individual = self.population[tb].__getRandomItems__()
                    
                    ob, _ = self.crossover(pb, pbb)
                    ob.skill_factor = tb

                    if self.eval_k[tb] >= self.max_eval_k[tb]:
                        ob.fcost = model.INF
                    else:
                        self.eval_k[tb] += 1
                        ob.fcost = self.tasks[tb](ob)

                    offsprings.__addIndividual__(ob)
                    
                    counter[tb] += 1
                    # self.eval_k[tb] += 1
                    
            stopping = sum(counter >= sub_size) == len(self.tasks)

        return offsprings

    def phaseTwo(self, D0):
        fcosts = [sub.getFitness() for sub in self.population]
        # start = time.time()
        D = self.calculateD(population = np.array([[ind.genes for ind in sub.ls_inds]for sub in self.population]), 
                            population_fitness = np.array(fcosts),
                            best = np.array([sub.__getBestIndividual__.genes for sub in self.population]),
                            )
        # end = time.time()
        # print("A: ", end - start)
        maxFit = np.max(fcosts, axis=1)
        minFit = np.min(fcosts, axis=1)
        maxDelta = maxFit - minFit + 1e-99

        assert len(D) == len(maxDelta), "Wrong shape. Got {} and {}".format(D.shape, maxDelta.shape)
        assert len(D) == len(self.tasks), "Got wrong shape"

        sigma = np.where(D > D0, 0, 1 - D/D0)
        nextPop = Population(IndClass = self.IndClass,
                            dim = self.dim_uss,
                            nb_inds_tasks=[0] * len(self.tasks),
                            list_tasks=self.tasks)
        # start = time.time()
        for i in range(len(self.tasks)):
            self.eval_k[i] += self.learningPhase[i].evolve(self.population[i], nextPop, sigma[i], maxDelta[i])
        # end = time.time()
        # print("B: ", end - start)
        self.population = nextPop

    def calculateD(self, population: np.array, population_fitness: np.array, best: np.array) -> np.array:
        '''
        Arguments include:\n
        + `population`: genes of the current population
        + `population_fitness`: fitness of the current population
        + `best`: the best gene of each subpop
        + `nb_tasks`: number of tasks
        '''
        
        D = np.empty((len(self.tasks)))
        for i in range(len(self.tasks)):
            gene_max = [np.max(population[i], axis = 1).tolist()] * self.dim_uss
            gene_min = [np.min(population[i], axis = 1).tolist()] * self.dim_uss

            D[i] = self.__class__._calculateD(np.array(gene_max).T, np.array(gene_min).T, population[i], population_fitness[i], best[i], model.TOLERANCE)
        return D
    
    @jit(nopython = True, parallel = True, cache=True)
    def _calculateD(gene_max: np.array, gene_min: np.array, subPop: np.array, subPop_fitness: np.array, best: np.array, TOLERANCE: float) -> float:
            # gene_max = gene_max.flatten()
            # gene_max = np.broadcast_to(gene_max, (subPop.shape[-1], subPop.shape[0])).T
            # gene_min = gene_min.flatten()
            # gene_min = np.broadcast_to(gene_min, (subPop.shape[-1], subPop.shape[0])).T
            
            w = np.where(subPop_fitness > TOLERANCE, 1/(subPop_fitness), 1/TOLERANCE)
            # w = [1/ind if ind > TOLERANCE else 1/TOLERANCE for ind in population[i]]
            # print(subPop.shape)
            sum_w = sum(w)
            d = (subPop - gene_min)/(gene_max - gene_min)
            best = (best - gene_min)/(gene_max - gene_min)
            d = np.sum((d - best) ** 2, axis=1)
            d = np.sqrt(d)
            assert d.shape == w.shape
            # d = np.sqrt(np.sum(d, axis=0))
            # d = np.sum([np.sqrt(np.sum((d[i] - best) * (d[i] - best))) for i in range(len(subPop))])

            return np.sum(w * d/sum_w)
    
    def updateRMP(self, c: int):
        for i in range(len(self.tasks)):
            for j in range(len(self.tasks)):
                if i == j:
                    continue
                if len(self.delta[i][j]) > 0:
                    self.rmp[i][j] += self.__class__._updateRMP(self.delta[i][j], self.s_rmp[i][j], c)
                else:
                    self.rmp[i][j] = (1 - c) * self.rmp[i][j]
                
                self.rmp[i][j] = max(0.1, min(1, self.rmp[i][j]))

    @jit(nopython = True, parallel = True, cache= True)
    def _updateRMP(delta: List, s_rmp: List, c: float) -> float:
        delta = np.array(delta)
        s_rmp = np.array(s_rmp)
        sum_delta = sum(delta)
        tmp = (delta/sum_delta) * s_rmp
        meanS = sum(tmp * s_rmp)
        
        return c * meanS/sum(tmp)
    
class LearningPhase():
    M = 2
    H = 10
    def __init__(self, IndClass, list_tasks, task) -> None:
        self.IndClass = IndClass
        self.list_tasks = list_tasks
        self.task = task
        self.sum_improv = [0.0] * LearningPhase.M
        self.consume_fes = [1.0] * LearningPhase.M
        self.mem_cr = [0.5] * LearningPhase.H
        self.mem_f = [0.5] * LearningPhase.H
        self.s_cr = []
        self.s_f = []
        self.diff_f = []
        self.mem_pos = 0
        self.gen = 0
        self.best_opcode = 1
        self.searcher = [self.pbest1, PolynomialMutation(nm = 5).getInforTasks(self.IndClass, self.list_tasks)]

    def evolve(self, subPop: SubPopulation, nextPop: Population, sigma: float, max_delta: float) -> SubPopulation:
        self.gen += 1

        if self.gen > 1:
            # start = time.time()
            self.best_opcode = self.__class__.updateOperator(sum_improve = self.sum_improv, 
                                                             consume_fes = self.consume_fes, 
                                                             M = LearningPhase.M)

            self.sum_improv = [0.0] * LearningPhase.M
            self.consume_fes = [1.0] * LearningPhase.M

            # end = time.time()
            # print("C: ", end - start)

        # self.updateMemory()
        
        pbest_size = max(5, int(0.15 * len(subPop)))
        # pbest = subPop.__getRandomItems__(size = pbest_size)
        idx_inds = np.argsort([ind.fcost for ind in subPop.ls_inds])
        pbest =  [subPop.ls_inds[i] for i in idx_inds[:pbest_size]] 
        # start1 = time.time()
        for ind in subPop:
            # start1 = time.time()
            # start = time.time()
            r = random.randint(0, LearningPhase.M - 1)
            cr = numba_random_gauss(self.mem_cr[r], 0.1)
            f = numba_random_cauchy(self.mem_f[r], 0.1)
            # end = time.time()
            # print("A: ", end - start)
            opcode = random.randint(0, LearningPhase.M)
            if opcode == LearningPhase.M:
                opcode = self.best_opcode

            self.consume_fes[opcode] += 1
            
            if opcode == 0:
                # start = time.time()
                child = self.searcher[opcode](ind, subPop, pbest, cr, f)
                # end = time.time()
                # print("C: ", end - start)
            elif opcode == 1:
                # start = time.time()
                child = self.searcher[opcode](ind, return_newInd=True)
                # end = time.time()
                # print("D: ", end - start)

            # start = time.time()
            child.skill_factor = ind.skill_factor
            child.fcost = self.task(child)
            
            diff = ind.fcost - child.fcost
            if diff > 0:
                survival = child

                self.sum_improv[opcode] += diff

                if opcode == 0:
                    self.diff_f.append(diff)
                    # self.s_cr.append(cr)
                    # self.s_f.append(f)
                
            elif diff == 0 or random.random() <= sigma * np.exp(diff/max_delta):
                survival = child
            else:
                survival = ind
            
            nextPop.__addIndividual__(survival)
            # end = time.time()
            # print("M: ", end - start)
        # end = time.time()
        # print("F: ", end - start1)
        return len(subPop)
    
    def pbest1(self, ind: Individual, subPop: SubPopulation, best: List[Individual], cr: float, f: float) -> Individual:
        pbest = best[random.randint(0, len(best) - 1)]
        
        ind_ran1, ind_ran2 = subPop.__getRandomItems__(size = 2, replace= False)
        
        u = (numba_random_uniform(len(ind.genes)) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (subPop.dim,))
            u[numba_randomchoice(subPop.dim)] = 1

        # new_genes = np.where(u, 
        #     pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
        #     ind.genes
        # )
        # # new_genes = np.clip(new_genes, ind.genes/2, (ind.genes + 1)/2)
        # new_genes = np.where(new_genes < 0, ind.genes/2, np.where(new_genes > 1, (ind.genes + 1)/2, new_genes))

        new_genes = self.__class__.produce_inds(ind.genes, pbest.genes, ind_ran1.genes, ind_ran2.genes, f, u)
        new_ind = self.IndClass(new_genes)

        return new_ind

    @jit(nopython=True, parallel = True)
    def produce_inds(ind_genes: np.array, best_genes: np.array, ind1_genes: np.array, ind2_genes: np.array, F: float, u: np.array) -> np.array:
        new_genes = np.where(u,
            best_genes + F * (ind1_genes - ind2_genes),
            ind_genes
        )
        new_genes = np.where(new_genes > 1, (ind_genes + 1)/2, new_genes) 
        new_genes = np.where(new_genes < 0, (ind_genes + 0)/2, new_genes)

        return new_genes

    # def updateMemory(self):
    #     if len(self.s_cr) > 0:
    #         # self.diff_f = np.array(self.diff_f)
    #         # self.s_cr = np.array(self.s_cr)
    #         # self.s_f = np.array(self.s_f)

    #         self.mem_cr[self.mem_pos] = self.__class__.updateMemoryCR(self.diff_f, self.s_cr)
    #         self.mem_f[self.mem_pos] = self.__class__.updateMemoryF(self.diff_f, self.s_f)
            
    #         self.mem_pos = (self.mem_pos + 1) % LearningPhase.H

    #         self.s_cr = []
    #         self.s_f = []
    #         self.diff_f = []

    # @jit(nopython = True, parallel = True, cache=True)
    # def updateMemoryCR(diff_f: List, s_cr: List) -> float:
    #     diff_f = np.array(diff_f)
    #     s_cr = np.array(s_cr)

    #     sum_diff = sum(diff_f)
    #     weight = diff_f/sum_diff
    #     tmp_sum_cr = sum(weight * s_cr)
    #     mem_cr = sum(weight * s_cr * s_cr)
        
    #     if tmp_sum_cr == 0 or mem_cr == -1:
    #         return -1
    #     else:
    #         return mem_cr/tmp_sum_cr
        
    # @jit(nopython = True, parallel = True, cache = True)
    # def updateMemoryF(diff_f: List, s_f: List) -> float:
    #     diff_f = np.array(diff_f)
    #     s_f = np.array(s_f)

    #     sum_diff = sum(diff_f)
    #     weight = diff_f/sum_diff
    #     tmp_sum_f = sum(weight * s_f)
    #     return sum(weight * (s_f ** 2)) / tmp_sum_f

    @jit(nopython = True, parallel = True)
    def updateOperator(sum_improve: List, consume_fes: List, M: int) -> int:
        sum_improve = np.array(sum_improve)
        consume_fes = np.array(consume_fes)
        eta = sum_improve / consume_fes
        best_rate = max(eta)
        best_op = np.argmax(eta)
        if best_rate > 0:
            return best_op
        else:
            return random.randint(0, M - 1)