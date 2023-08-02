from .compareModel import CompareModel, CompareModel_Original
from ..LoadSaveModel.load_utils import loadModel 

from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark


def render_cec_17_das_vs_wo_das(ls_model, ls_label, shape = (2, 5), min_cost= 1e-6, label_shape = None):
    '''Compare between S-MFEA with other MODEL'''

    # compare= CompareModel(
    #     models = [
            # loadModel("./Data/convergence_trend_das/S-MFEA_CEC17_PYTHON.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/S-MFEA_KL_CEC17_PYTHON.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MaTGA_w_DaS.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MaTGA_wo_DaS.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/SBSGA_w_DaS.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/SBSGA_wo_DaSCEC17.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MFEA_KL_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MFEA_SBX_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/LSA_KL_CEC17_PYTHON_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/LSA_SBX_CEC17_JAVA_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),

    #     ],
    #     label= [ '$\\bf{SM-MFEA}$', '$\\bf{SM-MFEA-DaS}$', "MaTGA-DaS", 'MaTGA', 'SBGA-DaS', 'SBGA', 'MFEA_DaS', 'MFEA', "LSA-DaS", "LSA",] 
    # )
    compare = CompareModel(
        models = ls_model, 
        label= ls_label
    )
    return compare.render(
        shape= shape, 
        min_cost= min_cost, 
        step=100, 
        yscale='log',
        title="",
        grid= True,
        showname= False,
        title_size= 20,
        label_size_x= 20,
        label_size_y= 20,
        x_tick_size= 20,
        y_tick_size= 20,
        handletextpad= 1,
        # borderaxespad=0.8,
        bbox_to_anchor=(0.5,-0.1),
        legend_size= 20,
        scatter_size=8,
        re= True,
        label_shape= label_shape if label_shape is not None else (2, len(ls_model) // 2 )
    )

def render_cec_17_SM_MFEA_vs_other(ls_model, ls_label, shape = (2, 5), min_cost= 1e-6, label_shape = None):
    '''Compare between S-MFEA with other MODEL'''

    # compare= CompareModel(
    #     models = [
            # loadModel("./Data/convergence_trend_das/S-MFEA_CEC17_PYTHON.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/S-MFEA_KL_CEC17_PYTHON.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MaTGA_w_DaS.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MaTGA_wo_DaS.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/SBSGA_w_DaS.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/SBSGA_wo_DaSCEC17.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MFEA_KL_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/MFEA_SBX_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/LSA_KL_CEC17_PYTHON_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),
            # loadModel("./Data/convergence_trend_das/LSA_SBX_CEC17_JAVA_1.mso",ls_tasks= CEC17_benchmark.get_10tasks_benchmark()[0]),

    #     ],
    #     label= [ '$\\bf{SM-MFEA}$', '$\\bf{SM-MFEA-DaS}$', "MaTGA-DaS", 'MaTGA', 'SBGA-DaS', 'SBGA', 'MFEA_DaS', 'MFEA', "LSA-DaS", "LSA",] 
    # )
    compare = CompareModel_Original(
        models = ls_model, 
        label= ls_label
    )
    return compare.render(
        shape= shape, 
        min_cost= min_cost, 
        step=100, 
        yscale='log',
        title="",
        grid= True,
        showname= False,
        title_size= 20,
        label_size_x= 20,
        label_size_y= 20,
        x_tick_size= 20,
        y_tick_size= 20,
        handletextpad= 1,
        # borderaxespad=0.8,
        bbox_to_anchor=(0.5,-0.1),
        legend_size= 20,
        scatter_size=8,
        re= True,
        label_shape= label_shape if label_shape is not None else (2, len(ls_model) // 2 )
    )




