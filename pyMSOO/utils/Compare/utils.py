from .compareModel import CompareModel 
from ..LoadSaveModel.load_utils import loadModel 

from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark


def render_cec_17(ls_model, ls_label, shape = (2, 5), min_cost= 1e-6):
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
    fig = compare.render(
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
        bbox_to_anchor=(0.5,-0.06),
        legend_size= 21,
        scatter_size=10,
        re= True
    )

    return fig

