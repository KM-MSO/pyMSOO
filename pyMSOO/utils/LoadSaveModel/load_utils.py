from pathlib import Path 
import pickle 
import os 

import pandas as pd 
import numpy as np 


def loadModel(PATH: str, ls_tasks=None, set_attribute=False):
    '''
    `.mso`
    '''
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name)):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only load model with .mso, not ' + \
            path_tmp.name[i:]

    f = open(PATH, 'rb')
    model = pickle.load(f)
    f.close()

    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    if model.tasks is None:
        model.tasks = ls_tasks
        if set_attribute is True:
            assert ls_tasks is not None, 'Pass ls_tasks plz!'
            model.compile_kwargs['tasks'] = ls_tasks
            for submodel in model.ls_model:
                submodel.tasks = ls_tasks
                submodel.last_pop.ls_tasks = ls_tasks
                for idx, subpop in enumerate(submodel.last_pop):
                    subpop.task = ls_tasks[idx]
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        setattr(getattr(submodel, attribute),
                                'tasks', ls_tasks)
                        pass
                else:
                    submodel.crossover.tasks = ls_tasks
                    submodel.mutation.tasks = ls_tasks

                # submodel.search.tasks = ls_tasks
                # submodel.crossover.tasks = ls_tasks
                # submodel.mutation.tasks = ls_tasks

    if model.name.split('.')[-1] == 'AbstractModel':
        model.name = path_tmp.name.split('.')[0]
    return model



def loadModelFromTxt(source_path, \
                     model_algorithm, \
                    multitime_model_class, 
                    target_path = "./", \
                    history_cost_shape= (1000, 2), \
                    nb_runs = 1, ls_tasks = [], 
                    remove_tasks: bool = False, \
                    name_model = None, 
                    total_time= None ):
    '''
    File txt has the format of MTO competition.

    Load result from file txt and save it to .mso file
    
    Args: 
        source_path: path file csv, txt, bla bla
        model: class of algorithms 
        multitime_model: class of multitime_model, not instance. avoid circular import 
        remove_tasks: Do remove tasks when save model or not ? 
        history_cost_shape: The history cost shape in one run. 
        nb_runs: the number run of model 
        ls_tasks: list of tasks
        name_model (optional): that will be use as name to save the model 
    
    Results: 
        Save multitime model file 
    '''

    assert len(ls_tasks) > 0 
    data = pd.read_csv(source_path, header= None) 
    # data = pd.read_csv(source_path, header= None, delim_whitespace= True).astype("float") 
    history_cost_shape = (data.shape[0], len(ls_tasks))
    nb_runs = (data.shape[1] - 1) // len(ls_tasks)
    history_cost = np.zeros(shape = (nb_runs,history_cost_shape[0], history_cost_shape[1]))
    data_transpose = data.transpose() 
    count_row = 1
    for i_run in range(nb_runs): 
        for idx_task in range(len(ls_tasks)):
            history_cost[i_run, :,idx_task] = data_transpose.iloc[count_row, :]
            count_row += 1  
    
    avg_history_cost = np.average(history_cost, axis = 0) 
    assert avg_history_cost.shape == history_cost_shape 


    multitime_model = multitime_model_class(model_algorithm) 
    multitime_model.__class__ = multitime_model_class
    multitime_model.compile()
    multitime_model.tasks = None 
    multitime_model.history_cost = avg_history_cost
    multitime_model.nb_run = nb_runs 

    for run in range(nb_runs): 
        new_model = model_algorithm.model() 
        new_model.history_cost = history_cost[run] 
        multitime_model.ls_model.append(new_model)
    if name_model is None:
        name_model = source_path.split("/")[-1].split(".")[0]
    
    if os.path.isdir(target_path) is False: 
        os.makedirs(target_path) 
    
    return multitime_model
    # return saveModel(model= mutiltime_model, PATH= f"{target_path}/{name_model}.mso", remove_tasks= remove_tasks, total_time= total_time) 

