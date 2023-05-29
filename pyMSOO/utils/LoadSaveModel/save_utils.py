from pathlib import Path 
import pickle 
import os 
import numpy as np 
import pandas as pd 

from .load_utils import loadModel, loadModelFromTxt
from pyMSOO.MFEA.model import AbstractModel 
from pyMSOO.MFEA.benchmark.continous import * 

def saveModel(model, PATH: str, remove_tasks=True, total_time = None ):
    '''
    `.mso`
    '''
    assert model.__class__.__name__ == 'MultiTimeModel'
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name) - 1, -1, -1):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only save model with .mso, not ' + \
            path_tmp.name[i:]

    tasks = model.tasks 

    if os.path.isfile(PATH): 
        import datetime
        now_time = datetime.datetime.now()
        ls_now_time = [now_time.year, now_time.month, now_time.day, now_time.hour, now_time.minute]
        str_now_time = "-".join([str(x) for x in ls_now_time])
        PATH = PATH[:-4] + str("-") + str(str_now_time) + '.mso'

    if total_time is not None: 
        model.total_time = total_time 

    if remove_tasks is True:
        if hasattr(model, "tasks"):
            model.tasks = None
        if hasattr(model, "compile_kwargs"):
            model.compile_kwargs['tasks'] = None
            for key, value in model.compile_kwargs.items():
                if hasattr(model.compile_kwargs[key], 'tasks'):
                    setattr(model.compile_kwargs[key], 'tasks', None)
        for submodel in model.ls_model:
            if hasattr(submodel, "tasks"):
                submodel.tasks = None
            if hasattr(submodel, "last_pop"):
                submodel.last_pop.ls_tasks = None
            if hasattr(submodel, "last_pop"):
                for subpop in submodel.last_pop:
                    subpop.task = None
            if hasattr(submodel, "kwargs"):
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        if hasattr(submodel, attribute): 
                            setattr(getattr(submodel, attribute), 'tasks', None)
                        pass
                else:
                    submodel.crossover.tasks = None
                    submodel.mutation.tasks = None

    try:
        f = open(PATH, 'wb')
        pickle.dump(model, f)
        f.close()

    except:
        cls = model.__class__
        model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

        if remove_tasks is True:
            if hasattr(model, "tasks"):
                model.tasks = tasks
            if hasattr(model, "compile_kwargs"): 
                model.compile_kwargs['tasks'] = None
            for submodel in model.ls_model:
                if hasattr(submodel, "tasks"):
                    submodel.tasks = tasks
                if hasattr(submodel, "last_pop"):
                    if hasattr(submodel.last_pop, "ls_tasks"):
                        submodel.last_pop.ls_tasks = tasks 
                    for idx, subpop in enumerate(submodel.last_pop):
                        if hasattr(subpop, "task"):
                            subpop.task = tasks[idx]
                if hasattr(submodel, "kwargs"):
                    if 'attr_tasks' in submodel.kwargs.keys():
                        for attribute in submodel.kwargs['attr_tasks']:
                            # setattr(submodel, getattr(subm, name), None)
                            setattr(getattr(submodel, attribute), 'tasks', tasks)
                            pass
                    else:
                        submodel.crossover.tasks = tasks
                        submodel.mutation.tasks = tasks 
        return 'Cannot Saved'

    
    if remove_tasks is True:
        if hasattr(model, "tasks"):
            model.tasks = tasks
        if hasattr(model, "compile_kwargs"): 
            model.compile_kwargs['tasks'] = None
        for submodel in model.ls_model:
            if hasattr(submodel, "tasks"):
                submodel.tasks = tasks
            if hasattr(submodel, "last_pop"):
                if hasattr(submodel.last_pop, "ls_tasks"):
                    submodel.last_pop.ls_tasks = tasks 
                
                for idx, subpop in enumerate(submodel.last_pop):
                    if hasattr(subpop, "task"):
                        subpop.task = tasks[idx]
            if hasattr(submodel, "kwargs"):
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        if hasattr(submodel, attribute): 
                            setattr(getattr(submodel, attribute), 'tasks', tasks)
                        pass
                else:
                    submodel.crossover.tasks = tasks
                    submodel.mutation.tasks = tasks 

    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    return 'Saved'
