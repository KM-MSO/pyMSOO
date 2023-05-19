from pathlib import Path 
import pickle 
import os 
import numpy as np 
import pandas as pd 

from .load_utils import loadModel, loadModelFromTxt
from pyMSOO.MFEA.model import AbstractModel 
from pyMSOO.utils.MultiRun.RunMultiTime import MultiTimeModel 
from pyMSOO.MFEA.benchmark.continous import * 
from .save_utils import saveModel

def export_history2cec_format(save_path= "./Complex/", \
                source_path = "./RESULTS/COMPLEX/SMP_v2/", \
                prefix_name = "MTOSOO_P",\
                ls_tasks =  WCCI22_benchmark.get_complex_benchmark(1)[0],
                total_evals = int(100000 * 2), 
                steps = 1000, 
                load_func = loadModel
                ):
    '''
    Export history cost of a all files in a folder to .txt files that has format of CEC competition
    '''
    for each_name_model in os.listdir(os.path.join(source_path)):
        model = load_func(os.path.join(source_path, each_name_model), ls_tasks) 
        
        tmp = np.concatenate([model.ls_model[i].history_cost for i in range(len(model.ls_model))], axis=1)
        stt = np.arange(total_evals // steps, total_evals + total_evals // steps, total_evals // steps)
        # stt = stt.reshape(1,int((200000)/ 2000))
        stt = stt.reshape(1, -1) 
        tmp = tmp.T
        if tmp.shape[0] != stt.shape[1]:
            tmp = tmp[:, -stt.shape[1]:]
        
        assert tmp.shape[1] == stt.shape[1], print(tmp.shape)
        tmp4= np.concatenate([stt, tmp], axis=0).T

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        f = open(save_path + prefix_name+ each_name_model.split(".")[0] + ".txt", "w")
        for line in tmp4: 
            f.write(", ".join([str(int(line[0]))] + [str('{:.6f}'.format(e)) for e in line[1:]]) + "\n")
        f.close()
    

def export_solution2csv(folder_path,save_name_csv= "history_cost_summaries.csv", name= [str(i) + ".mso" for i in range(1, 11)], min_value= 1e-6): 
    '''
    export solution of algorithms to csv file with each column correspond an algorithm
    
    Arg: 
        folder_path: folder contain algorithms folder
                    Ex: Result/ contain three algorithms MFEA, SM, LSA 
                    Ex: Result/ contains MFEA/, SM/, LSA/ 
                    Ex:  MFEA/, SM/, LSA/ contain file .mso 
        name: name file mso in each algorithms folder. They must be same. 
        min_value: if "result" is smaller min_value => change to 0 
    '''
    history_cost_summaries = []
    for model_lib in os.listdir(folder_path): 
        print(model_lib)
        tmp = []
        # for id in os.listdir("RESULTS/SM_MFEA_DaS/WCCI22/SM_MFEA_DaS/"):
        for id in name:
            print("id: ", id)
            model = loadModel(os.path.join(os.path.join(folder_path, model_lib), id), ls_tasks= WCCI22_benchmark.get_50tasks_benchmark(1)[0])
            tmp += np.array(model.history_cost[-1]).tolist() 
        print(len(tmp))
        history_cost_summaries.append(tmp.copy())
        

    ls_col_name = os.listdir(folder_path)
    history_cost_summaries = np.array(history_cost_summaries)
    history_cost_summaries = np.where(history_cost_summaries < min_value, 0, history_cost_summaries)
    print(history_cost_summaries.T.shape)
    df = pd.DataFrame(history_cost_summaries.T,columns= ls_col_name)
    pd.DataFrame.to_csv(df, save_name_csv)
    pass 


import pickle 
import os 

def export2txt(source_file, 
               save_file, 
               total_evals = int(100000 * 2), 
                steps = 1000, ):
        model = pickle.load(open(source_file, 'rb'))
        tmp = np.concatenate([model['ls_model'][i]['history_cost'] for i in range(len(model['ls_model']))], axis=1)
        stt = np.arange(total_evals // steps, total_evals + total_evals // steps, total_evals // steps)
        # stt = stt.reshape(1,int((200000)/ 2000))
        stt = stt.reshape(1, -1) 
        tmp = tmp.T
        if tmp.shape[1] > stt.shape[1]:
            tmp = tmp[:, -stt.shape[1]:]
        else: 
            print(tmp.shape)
            print(tmp[0:2, 0])
            tmp = np.concatenate([np.repeat(tmp[:, 0:1], stt.shape[1] - tmp.shape[1], axis=1), tmp], axis=1)
            print(tmp.shape)
            print(tmp[0:2, 1])
        assert tmp.shape[1] == stt.shape[1], print(tmp.shape)
        tmp4= np.concatenate([stt, tmp], axis=0).T

        f = open(save_file, "w")
        for line in tmp4: 
            f.write(", ".join([str(int(line[0]))] + [str('{:.6f}'.format(e)) for e in line[1:]]) + "\n")
        f.close()

def export_all2txt(source_folder, destination_folder): 
    for item in os.listdir(source_folder): 
        current_source_path= os.path.join(source_folder, item)
        if os.path.isdir(current_source_path):
            export_all2txt(current_source_path, os.path.join(destination_folder, item))
        else: 
            current_save_path = os.path.join(destination_folder, item)[:-4] + ".txt"
            if os.path.isdir(destination_folder) is False: 
                os.makedirs(destination_folder) 
            print(current_source_path)
            # file = pickle.load(open(current_source_path, 'rb'))
            export2txt(
                 source_file= current_source_path, 
                 save_file= current_save_path, 
                 total_evals= int(1e5 * 50), 
                 steps= 1000, 
            )

def export_all2mso(source_folder, destination_folder): 
    for item in os.listdir(source_folder): 
        current_source_path= os.path.join(source_folder, item)
        if os.path.isdir(current_source_path):
            export_all2mso(current_source_path, os.path.join(destination_folder, item))
        else: 
            current_save_path = os.path.join(destination_folder, item)[:-4] + ".mso"
            if os.path.isdir(destination_folder) is False: 
                os.makedirs(destination_folder) 
            # print(current_source_path)
            # file = pickle.load(open(current_source_path, 'rb'))
            model = loadModelFromTxt(
                source_path= current_source_path, 
                model_algorithm= AbstractModel, 
                multitime_model_class= MultiTimeModel, 
                target_path= "./", 
                history_cost_shape= (1000, 50), 
                nb_runs= 30, 
                ls_tasks= WCCI22_benchmark.get_50tasks_benchmark(1)[0] 
            )
            saveModel(
                model= model, 
                PATH= current_save_path, 
                remove_tasks= True, 
            )
