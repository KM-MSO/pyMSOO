import pickle
import numpy as np
import traceback
import os
from typing import List
from pathlib import Path

from ...MFEA.model import AbstractModel
from ..LoadSaveModel.save_utils import saveModel

def get_model_name(model: AbstractModel.model):
    fullname = model.__module__
    index = None
    for i in range(len(fullname) - 1, -1, -1):
        if fullname[i] == '.':
            index = i
            break
    if index is None:
        return fullname
    return fullname[index + 1:]


class MultiTimeModel:
    def __init__(self, model: AbstractModel = None, list_attri_avg: list = None,  name=None) -> None:
        if model is not None:
            if type(model).__name__ == 'module':
                self.model = model.model 
            elif type(model).__name__ == 'type': 
                self.model = model 
        
            if name is None and model is not None:
                self.name = model.__name__
            else:
                self.name = name
            self.model = model.model 

            if list_attri_avg is None:
                self.list_attri_avg = None
            else:
                self.list_attri_avg = list_attri_avg

            self.ls_model: List[AbstractModel.model] = []
            self.ls_seed: List[int] = []
            self.total_time = 0

            # add inherit
            cls = self.__class__
            self.__class__ = cls.__class__(cls.__name__, (cls, self.model), {})

            # status of model run
            # self.status = 'NotRun' | 'Running' | 'Done'
            self.status = 'NotRun'

    def set_data(self, history_cost: np.ndarray):
        self.status = 'Done'
        self.history_cost = history_cost
        print('Set complete!')

    def set_attribute(self):
        # print avg
        if self.list_attri_avg is None:
            self.list_attri_avg = self.ls_model[0].ls_attr_avg
        for i in range(len(self.list_attri_avg)):
            try:
                result = [model.__getattribute__(
                    self.list_attri_avg[i]) for model in self.ls_model]
            except:
                print("cannot get attribute {}".format(self.list_attri_avg[i]))
                continue
            try:
                
            # min_dim1 = 1e10 
            # for idx1, array_seed in enumerate(result): 
            #     min_dim1 = min([len(array_seed), min_dim1])
            #     for idx2,k in enumerate(array_seed): 
                        
            #         for idx3, x in enumerate(k) : 
            #             if type(x) != float:
            #                 result[idx1][idx2][idx3] = float(x)
                    
            #         result[idx1][idx2]= np.array(result[idx1][idx2])
            #     result[idx1] = np.array(result[idx1])
            
            # for idx, array in enumerate(result):
            #     result[idx] = result[idx][:min_dim1]
            
                result = np.array(result[:][:min([len(his) for his in result])][:])
                result = np.average(result, axis=0)
                self.__setattr__(self.list_attri_avg[i], result)
            except:
                print(f'can not convert {self.list_attri_avg[i]} to np.array')
                continue

    def print_result(self, print_attr = [], print_time = True, print_name= True):
        # print time
        seconds = self.total_time
        minutes = seconds // 60
        seconds = seconds - minutes * 60
        if print_time: 
            print("total time: %02dm %.02fs" % (minutes, seconds))

        # print avg
        if self.list_attri_avg is None:
            self.list_attri_avg = self.ls_model[0].ls_attr_avg

        if len(print_attr) ==0 : 
            print_attr = self.list_attri_avg 
        
        for i in range(len(print_attr)):
            try:
                result = self.__getattribute__(print_attr[i])[-1]
                if print_name: 
                    print(f"{print_attr[i]} avg: ")
                np.set_printoptions(
                    formatter={'float': lambda x: format(x, '.2E')})
                print(result)
            except:
                try:
                    result = [model.__getattribute__(
                        print_attr[i]) for model in self.ls_model]
                    result = np.array(result)
                    result = np.sum(result, axis=0) / len(self.ls_model)
                    if print_name: 
                        print(f"{print_attr[i]} avg: ")
                    np.set_printoptions(
                        formatter={'float': lambda x: format(x, '.2E')})
                    print(result)
                except:
                    print(
                        f'can not convert {print_attr[i]} to np.array')

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def run(self, nb_run: int = None, save_path: str = "./RESULTS/result/model.mso", seed_arr: list = None, random_seed: bool = False, replace_folder = True):
        print('Checking ...', end='\r')
        # folder
        idx = len(save_path) 
        while save_path[idx-1] != "/": 
            idx -= 1 

        if os.path.isdir(save_path[:idx]) is True:
            if replace_folder is True:
                pass
            else:
                raise Exception("Folder is existed")
        else:
            os.makedirs(save_path[:idx])

        if self.status == 'NotRun':
            if nb_run is None:
                self.nb_run = 1
            else:
                self.nb_run = nb_run

            if save_path is None:
                save_path = get_model_name(self.model) + '.mso'

            if seed_arr is not None:
                assert len(seed_arr) == nb_run
            elif random_seed:
                seed_arr = np.random.randint(
                    nb_run * 100, size=(nb_run, )).tolist()
            else:
                seed_arr = np.arange(nb_run).tolist()

            self.ls_seed = seed_arr
            index_start = 0
        elif self.status == 'Running':
            if nb_run is not None:
                assert self.nb_run == nb_run

            if save_path is None:
                save_path = get_model_name(self.model) + '.mso'

            if seed_arr is not None:
                assert np.all(
                    seed_arr == self.ls_seed), '`seed_arr` is not like `ls_seed`'

            index_start = len(self.ls_model)
        elif self.status == 'Done':
            print('Model has already completed before.')
            return
        else:
            raise ValueError('self.status is not NotRun | Running | Done')

        for idx_seed in range(index_start, len(self.ls_seed)):
            try:
                model = self.model(self.ls_seed[idx_seed])
                
                self.ls_model.append(model)
                
                model.compile(**self.compile_kwargs)
                model.fit(*self.args, **self.kwargs)

                self.total_time += model.time_end - model.time_begin

            except KeyboardInterrupt as e:
                self.status = 'Running'
                # self.set_attribute()

                # self.__class__ = MultiTimeModel
                # save_result = saveModel(self, save_path)
                print('\n\nKeyboardInterrupt: ' +
                      save_result + ' model, model is not Done')
                traceback.print_exc()
                break
            try:
                
                temp_class = self.__class__ 
                self.__class__ = MultiTimeModel
                save_result = saveModel(self, save_path, remove_tasks= False)
                self.__class__ = temp_class 
                print("saved ", idx_seed)
            except:
                print("Cannot save")
        else:
            self.set_attribute()
            self.status = 'Done'
            print('DONE!')
            self.__class__ = MultiTimeModel
            print(saveModel(self, save_path))


