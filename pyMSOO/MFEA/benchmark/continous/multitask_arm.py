import os
from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Tuple, Type

import numpy as np
from numba import jit, njit
from numba.typed import Dict, List
from sklearn.cluster import KMeans

from ....utils.EA import AbstractTask, Individual
from ..continous.funcs import AbstractFunc
from .utils import Individual_func


class MultiArmBenchmark():
    def __init__(self, 
                 num_task = 10,
                 dim_map = 2, 
                 dim = 10,
                 samples = 30000, 
                 cvt_use_cache=True,
                 save_dir = './Data') -> None:
        """Create a new dataset for robot arm problem

        Args:
            num_task (int, optional): Number of task. Defaults to 10.
            dim_map (int, optional): The dimension of the problem. Defaults to 2.
            dim (int, optional): The dimension of each individual. Defaults to 10.
            samples (int, optional): Number of random samples. Defaults to 30000.
            cvt_use_cache (bool, optional): Generate new datasets or not. Defaults to True.
            save_dir (str, optional): Save directory. Defaults to './Data'.
        """

        self.num_task = num_task
        self.dim_map = dim_map
        self.dim = dim
        self.samples = samples
        self.save_dir = save_dir
        self.cvt_use_cache = cvt_use_cache
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Directory created:", save_dir)
        else:
            print("Directory already exists:", save_dir)
            
        self.tasks = self.cvt()

    def __centroids_filename(self, k, dim):
        return 'centroids_' + str(k) + '_' + str(dim) + '.dat'

    def __write_centroids(self, centroids):
        k = centroids.shape[0]
        dim = centroids.shape[1]
        filename = os.path.join(self.save_dir, self.__centroids_filename(k, dim))
        with open(filename, 'w') as f:
            for p in centroids:
                for item in p:
                    f.write(str(item) + ' ')
                f.write('\n')

    def cvt(self):
        # check if we have cached values
        cvt_use_cache = self.cvt_use_cache
        fname = os.path.join(self.save_dir, self.__centroids_filename(self.num_task, self.dim_map))
        
        if cvt_use_cache:
            if Path(fname).is_file():
                print("WARNING: using cached CVT:", fname)
                return np.loadtxt(fname)
        # otherwise, compute cvt
        print("Computing CVT (this can take a while...):", fname)

        x = np.random.rand(self.samples, self.dim_map)
        k_means = KMeans(init='k-means++', n_clusters=self.num_task,
                        n_init=1, verbose=1)#,algorithm="full")
        k_means.fit(x)
        self.__write_centroids(k_means.cluster_centers_)

        return k_means.cluster_centers_

    def getMultiTask(self)-> Tuple[List[AbstractFunc], Type[Individual_func]]:
        tasks = [
            Arm(self.dim, task[0], task[1], (0, 1)) for task in self.tasks
        ]
        return tasks, Individual_func


class Arm(AbstractTask):
    def __init__(self, dim: int, alpha_max: float, d: int, bound: tuple = None):
        """Robot arm problem

        Args:
            dim (int): The dimension of individual
            alpha_max (float): The sum of maximum angle
            d (int): The sum of arm length
            bound (tuple, optional): Bound of gene. Defaults to None.
        """

        self.dim = dim
        self.bound = bound
        self.angular_range = alpha_max/dim
        self.d = d
        self.lengths = np.ones(dim) * d / dim
        self.lengths = np.concatenate(([0], self.lengths))

    def fw_kinematics(self, p):
        # ONLY USE THIS FUNCTION FOR TESTING PURPOSE
        assert(len(p) == self.dim)
        p = np.append(p, 0)
        return self.__class__._func(self.dim, self.lengths, p)

    # @staticmethod
    @njit
    def decode(x, angular_range):
        x_decode = (x - 0.5) * angular_range * pi * 2
        return x_decode
    
    @staticmethod
    def _convert(x):
        if isinstance(x, np.ndarray):
            return x

        if isinstance(x, Individual):
            return x.genes

        raise ValueError(
            "Wrong value type for input argument, expected 'List', 'np.ndarray' or 'Individual' but got {}".format(type(x)))

    def __call__(self, x):
        x = self.__class__._convert(x)
        x = self.__class__.decode(x, self.angular_range)
        return self.func(x)
    
    def func(self, p: List):
        assert(len(p) == self.dim)
        p = np.append(p, 0)
        y = self.__class__._func(self.dim, self.lengths, p)
        target = 0.5 * np.ones(2)
        f = np.linalg.norm(y - target)
        return f
    
    # @staticmethod
    @njit
    def _func(dim, lengths, p):
        def dot_py(A,B):
            m, n = A.shape
            p = B.shape[1]

            C = np.zeros((m,p))

            for i in range(0,m):
                for j in range(0,p):
                    for k in range(0,n):
                        C[i,j] += A[i,k]*B[k,j] 
            return C
        
        mat = np.identity(4)
        for i in range(0, dim + 1):   
            m = np.array([cos(p[i]), -sin(p[i]), 0, lengths[i], \
                         sin(p[i]),  cos(p[i]), 0, 0, \
                         0, 0, 1, 0, 0, 0, 0, 1])
            m = m.reshape((4, 4))
            mat = dot_py(mat, m)
            v = dot_py(mat, np.array([0, 0, 0, 1]).reshape((4, 1)))
            # mat = np.dot(mat, m)
            # v = np.dot(mat, np.array([0.0, 0.0, 0.0, 1.0]).reshape((4, 1)))
        return [v[0][0], v[0][1]]

if __name__ == "__main__":
    # 1-DOFs
    # a = Arm([1])
    # v,_ = a.fw_kinematics([0])
    # np.testing.assert_almost_equal(v, [1, 0])
    # v,_ = a.fw_kinematics([pi/2])
    # np.testing.assert_almost_equal(v, [0, 1])

    # # 2-DOFs
    # a = Arm([1, 1])
    # v,_ = a.fw_kinematics([0, 0])
    # np.testing.assert_almost_equal(v, [2, 0])
    # v,_ = a.fw_kinematics([pi/2, 0])
    # np.testing.assert_almost_equal(v, [0, 2])
    # v,_ = a.fw_kinematics([pi/2, pi/2])
    # np.testing.assert_almost_equal(v, [-1, 1])
    # v,x = a.fw_kinematics([pi/4, -pi/2])
    # np.testing.assert_almost_equal(v, [sqrt(2), 0])

    # a 4-DOF square
    # a = Arm([1, 1, 1,1])
    # v,_ = a.fw_kinematics([pi/2, pi/2, pi/2, pi/2])
    # np.testing.assert_almost_equal(v, [0, 0])
    # 1-DOFs
    a = Arm(dim = 1, alpha_max=1, d=1, bound=(0, 1))
    v = a.fw_kinematics(np.array([0]))
    np.testing.assert_almost_equal(v, [1, 0])
    v = a.fw_kinematics(np.array([pi/2]))
    np.testing.assert_almost_equal(v, [0, 1])

    # 2-DOFs
    a = Arm(dim = 2, alpha_max=1, d=2, bound=(0, 1))
    v = a.fw_kinematics([0, 0])
    np.testing.assert_almost_equal(v, [2, 0])
    v = a.fw_kinematics([pi/2, 0])
    np.testing.assert_almost_equal(v, [0, 2])
    v = a.fw_kinematics([pi/2, pi/2])
    np.testing.assert_almost_equal(v, [-1, 1])
    v = a.fw_kinematics([pi/4, -pi/2])
    np.testing.assert_almost_equal(v, [sqrt(2), 0])

    # a 4-DOF square
    a = Arm(dim = 4, alpha_max=1, d=4, bound=(0, 1))
    v = a.fw_kinematics([3*pi/4, pi/2, pi/2, pi/2])
    np.testing.assert_almost_equal(v, [0, 0])