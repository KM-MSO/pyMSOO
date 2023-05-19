import numpy as np
import random
from numba import jit

from . import AbstractModel
from ...utils import Crossover, Mutation, Selection, DimensionAwareStrategy
from ...utils.Mutation import GaussMutation
from ...utils.EA import *
from ...utils.numba_utils import numba_randomchoice, numba_random_gauss, numba_random_cauchy, numba_random_uniform
from ...utils.Search import *

class model(AbstractModel.model):
    TOLERANCE = 1e-6
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
        np.fill_diagonal(self.rmp, 0)

        # self.delta = [[[] for _ in range(len(self.tasks))] for _ in range(len(self.tasks))]

        # self.s_rmp = [[[] for _ in range(len(self.tasks))] for _ in range(len(self.tasks))]

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
        self.eval_k = [0] * len(self.tasks)
        epoch = 1
        
        D0 = self.calculateD(population = np.array([[ind.genes for ind in sub.ls_inds] for sub in self.population]), 
                            population_fitness = np.array([sub.getFitness() for sub in self.population]),
                            best = np.array([sub.__getBestIndividual__.genes for sub in self.population]),)

        while sum(self.eval_k) < MAXEVALS:
            self.delta = [[[] for _ in range(len(self.tasks))] for _ in range(len(self.tasks))]

            self.s_rmp = [[[] for _ in range(len(self.tasks))] for _ in range(len(self.tasks))]

            offsprings = self.reproduction(sum(self.nb_inds_tasks), self.population)
            
            # merge and update rank
            self.population = self.population + offsprings
            self.population.update_rank()
            
            # selection
            self.nb_inds_tasks = [int(
                int(max((nb_inds_min - nb_inds_max) * (sum(self.eval_k)/MAXEVALS) + nb_inds_max, nb_inds_min))
            )] * len(self.tasks)
            self.selection(self.population, self.nb_inds_tasks)

            # update operators
            self.crossover.update(population = self.population)
            self.mutation.update(population = self.population)
            self.dimension_strategy.update(population = self.population)

            self.updateRMP(c)

            # self.phaseTwo(D0)

            if sum(self.eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                # save history
                self.history_cost.append([ind.fcost for ind in self.population.get_solves()])
                self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[sum(self.nb_inds_tasks)], self.history_cost[-1]], use_sys= True)
                epoch += 1

        print('\nEND!')

        #solve 
        self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[sum(self.nb_inds_tasks)], self.history_cost[-1]], use_sys= True)
        self.last_pop = self.population
        return self.last_pop.get_solves()

    def phaseTwo(self, D0):
        fcosts = [sub.getFitness() for sub in self.population]

        D = self.calculateD(population = np.array([[ind.genes for ind in sub.ls_inds]for sub in self.population]), 
                            population_fitness = np.array(fcosts),
                            best = np.array([sub.__getBestIndividual__.genes for sub in self.population]),
                            )
        
        maxFit = np.max(fcosts, axis=0)
        minFit = np.min(fcosts, axis=0)
        maxDelta = maxFit - minFit + 1e-99

        sigma = np.where(D > D0, 0, 1 - D/D0)
        nextPop = Population(IndClass = self.IndClass,
                        dim = self.dim_uss,
                        nb_inds_tasks=[0] * len(self.tasks),
                        list_tasks=self.tasks)
        # best = self.learningPhase.evolve(self.population, sigma, maxDelta)
        # self.population += best
        for i in range(len(self.tasks)):
            nextPop, tmp_eval = self.learningPhase[i].evolve(self.population[i], nextPop, sigma[i], maxDelta[i])
            self.eval_k[i] += tmp_eval

        self.population += nextPop
    
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
                self.eval_k[ta] += 2

                oa, ob = self.crossover(pa, pb)

                oa.skill_factor = ta
                ob.skill_factor = ta
                
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)

                counter[ta] += 2

            elif random.random() <= rmpValue:
                off = self.crossover(pa, pb)

                for o in off:
                    if counter[ta] < sub_size and random.random() < self.rmp[ta][tb]/(self.rmp[ta][tb] + self.rmp[tb][ta]):
                        o.skill_factor = ta
                        o = self.dimension_strategy(o, tb, pa)
                        o.fcost = self.tasks[ta](o)

                        offsprings.__addIndividual__(o)
                        
                        counter[ta] += 1
                        self.eval_k[ta] += 1
                        
                        if pa.fcost > o.fcost:
                            self.delta[ta][tb].append(pa.fcost - o.fcost)
                            self.s_rmp[ta][tb].append(rmpValue)
                    
                    elif counter[tb] < sub_size:
                        o.skill_factor = tb
                        o = self.dimension_strategy(o, ta, pb)
                        o.fcost = self.tasks[tb](o)

                        offsprings.__addIndividual__(o)
                        
                        counter[tb] += 1
                        self.eval_k[tb] += 1

                        if pb.fcost > o.fcost:
                            self.delta[tb][ta].append(pb.fcost - o.fcost)
                            self.s_rmp[tb][ta].append(rmpValue)

            else:
                if counter[ta] < sub_size:
                    paa = self.population[ta].__getRandomItems__(1)[0]
                    
                    oa, _ = self.crossover(pa, paa)
                    oa.skill_factor = ta

                    offsprings.__addIndividual__(oa)

                    counter[ta] += 1
                    self.eval_k[ta] += 1

                if counter[tb] < sub_size:
                    pbb = self.population[tb].__getRandomItems__(1)[0]
                    
                    ob, _ = self.crossover(pb, pbb)
                    ob.skill_factor = tb
                    
                    offsprings.__addIndividual__(ob)
                    
                    counter[tb] += 1
                    self.eval_k[tb] += 1
                    
            stopping = sum(counter >= sub_size) == len(self.tasks)

        return offsprings

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
            gene_max = np.max(population[i], axis = 0)
            gene_min = np.min(population[i], axis = 0)

            D[i] = self.__class__._calculateD(gene_max, gene_min, population[i], population_fitness[i], best[i], model.TOLERANCE)
        return D
    
    @jit(nopython = True, parallel = True, cache=True)
    def _calculateD(gene_max: np.array, gene_min: np.array, subPop: np.array, subPop_fitness: np.array, best: np.array, TOLERANCE: float) -> float:
            w = np.where(subPop_fitness > TOLERANCE, 1/(subPop_fitness), 1/TOLERANCE)
            # w = [1/ind if ind > TOLERANCE else 1/TOLERANCE for ind in population[i]]
            sum_w = sum(w)
            d = (subPop - gene_min)/(gene_max - gene_min)
            best = (best - gene_min)/(gene_max - gene_min)
            d = np.sum(np.sqrt((d - best) * (d - best)))

            return np.sum(w/sum_w * d)
    
    def updateRMP(self, c: int):
        for i in range(len(self.tasks)):
            for j in range(len(self.tasks)):
                if i == j:
                    continue
                if len(self.delta[i][j]) > 0:
                    self.rmp[i][j] += self.__class__._updateRMP(self.delta[i][j], self.s_rmp[i][j], c)
                # if len(delta[i][j]) > 0:
                #     delta[i][j] = np.array(delta[i][j])
                #     s_rmp[i][j] = np.array(s_rmp[i][j])

                #     sum_delta = sum(delta[i][j])
                #     meanS = sum((delta[i][j]/sum_delta) * s_rmp[i][j] * s_rmp[i][j])
                #     sum_s_rmp = sum((delta[i][j]/sum_delta) * s_rmp[i][j])
                #     tmp_sum = meanS/sum_s_rmp
                    
                #     rmp[i][j] += c * meanS/tmp_sum
                else:
                    self.rmp[i][j] = (1 - c) * self.rmp[i][j]
                
                self.rmp[i][j] = max(0.1, min(1, self.rmp[i][j]))

    @jit(nopython = True, parallel = True, cache=True)
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
        # self.sum_improv = np.zeros((LearningPhase.M))
        # self.consume_fes = np.ones((LearningPhase.M))
        # self.mem_cr = np.full((LearningPhase.H), 0.5)
        # self.mem_f = np.full((LearningPhase.H), 0.5)
        self.sum_improv = [0] * LearningPhase.M
        self.consume_fes = [0] * LearningPhase.M
        self.mem_cr = [0.5] * LearningPhase.H
        self.mem_f = [0.5] * LearningPhase.H
        self.s_cr = []
        self.s_f = []
        self.diff_f = []
        self.mem_pos = 0
        self.gen = 0
        self.best_opcode = 1
        self.searcher = [self.pbest1, GaussMutation().getInforTasks(self.IndClass, self.list_tasks)]

    # def evolve(self, population: Population, sigma: np.array, max_delta: np.array):
    #     nextPop = Population(IndClass = self.IndClass,
    #                          dim = population.dim_uss,
    #                          nb_inds_tasks=[0] * len(self.list_tasks),
    #                          list_tasks=self.list_tasks)
    #     for t in range(len(self.list_tasks)):
    #         nextPop = self._evolve(population[t], nextPop, sigma[t], max_delta[t])

    #     return nextPop

    def evolve(self, subPop: SubPopulation, nextPop: Population, sigma: float, max_delta: float) -> SubPopulation:
        eval_k = 0
        
        self.gen += 1
        if self.gen > 1:
            self.best_opcode = self.__class__.updateOperator(sum_improve = self.sum_improv, 
                                                             consume_fes = self.consume_fes, 
                                                             M = LearningPhase.M)

            # self.sum_improv = np.zeros((LearningPhase.M))
            # self.consume_fes = np.ones((LearningPhase.M))
            self.sum_improv = [0.0] * LearningPhase.M
            self.consume_fes = [1.0] * LearningPhase.M

        # self.updateMemory()
        
        pbest_size = max(5, int(0.15 * len(subPop)))
        pbest = subPop.__getRandomItems__(size = pbest_size)

        for ind in subPop:
            r = random.randint(0, LearningPhase.M - 1)
            cr = numba_random_gauss(self.mem_cr[r], 0.1)
            f = numba_random_cauchy(self.mem_f[r], 0.1)
                        
            opcode = random.randint(0, LearningPhase.M)
            if opcode == LearningPhase.M:
                opcode = self.best_opcode
            
            if opcode == 0:
                child = self.searcher[opcode](ind, subPop, pbest, cr, f)
            elif opcode == 1:
                child = self.searcher[opcode](ind, return_newInd=True)

            child.skill_factor = ind.skill_factor
            child.fcost = self.task(child)
            eval_k += 1
            
            diff = ind.fcost - child.fcost
            if diff > 0:
                survival = child

                self.sum_improv[opcode] += diff

                if opcode == 0:
                    self.diff_f.append(diff)
                    self.s_cr.append(cr)
                    self.s_f.append(f)
                
            elif diff == 0 or random.random() <= sigma * np.exp(diff/max_delta):
                survival = child
            else:
                survival = ind
            
            nextPop.__addIndividual__(survival)
        
        return nextPop, eval_k
    
    def pbest1(self, ind: Individual, subPop: SubPopulation, best: List[Individual], cr: float, f: float) -> Individual:
        pbest = best[random.randint(0, len(best) - 1)]
        
        ind_ran1, ind_ran2 = subPop.__getRandomItems__(size = 2, replace= False)
        
        u = (numba_random_uniform(len(ind.genes)) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (subPop.dim,))
            u[numba_randomchoice(subPop.dim)] = 1

        new_genes = np.where(u, 
            pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        # new_genes = np.clip(new_genes, ind.genes/2, (ind.genes + 1)/2)
        new_genes = np.where(new_genes < 0, ind.genes/2, np.where(new_genes > 1, (ind.genes + 1)/2, new_genes))
        new_ind = self.IndClass(new_genes)

        return new_ind

    # def updateMemory(self):
    #     if len(self.s_cr) > 0:
    #         self.s_cr = np.array(self.s_cr)
    #         self.s_f = np.array(self.s_f)
    #         self.diff_f = np.array(self.diff_f)

    #         sum_dff = sum(self.diff_f)
    #         weight = np.array(self.diff_f)/sum_dff
    #         self.mem_f[self.mem_pos] = sum(weight * self.s_f * self.s_f)
    #         self.mem_cr[self.mem_pos] = sum(weight * self.s_cr * self.s_cr)

    #         tmp_sum_f = sum(weight * self.s_f)
    #         tmp_sum_cr = sum(weight * self.s_cr)

    #         self.mem_f[self.mem_pos] /= tmp_sum_f

    #         if tmp_sum_cr == 0 or self.mem_cr[self.mem_pos] == -1:
    #             self.mem_cr[self.mem_pos] = -1
    #         else:
    #             self.mem_cr[self.mem_pos] /= tmp_sum_cr

    #         self.mem_pos += 1
    #         if self.mem_pos >= LearningPhase.H:
    #             self.mem_pos = 0
            
    #         self.s_cr = []
    #         self.s_f = []
    #         self.diff_f = []

    def updateMemory(self):
        if len(self.s_cr) > 0:
            # self.diff_f = np.array(self.diff_f)
            # self.s_cr = np.array(self.s_cr)
            # self.s_f = np.array(self.s_f)

            self.mem_cr[self.mem_pos] = self.__class__.updateMemoryCR(self.diff_f, self.s_cr)
            self.mem_f[self.mem_pos] = self.__class__.updateMemoryF(self.diff_f, self.s_f)
            
            self.mem_pos = (self.mem_pos + 1) % LearningPhase.H

            self.s_cr = []
            self.s_f = []
            self.diff_f = []

    @jit(nopython = True, parallel = True, cache=True)
    def updateMemoryCR(diff_f: List, s_cr: List) -> float:
        diff_f = np.array(diff_f)
        s_cr = np.array(s_cr)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tmp_sum_cr = sum(weight * s_cr)
        mem_cr = sum(weight * s_cr * s_cr)
        
        if tmp_sum_cr == 0 or mem_cr == -1:
            return -1
        else:
            return mem_cr/tmp_sum_cr
        
    @jit(nopython = True, parallel = True, cache=True)
    def updateMemoryF(diff_f: List, s_f: List) -> float:
        diff_f = np.array(diff_f)
        s_f = np.array(s_f)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tmp_sum_f = sum(weight * s_f)
        return sum(weight * (s_f ** 2)) / tmp_sum_f

    @jit(nopython = True, parallel = True, cache=True)
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