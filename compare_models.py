from pyMSOO.utils.Compare.compareModel import *
from pyMSOO.utils.LoadSaveModel import *



import os, yaml

with open("RESULTS/src.yml", "r") as yamlfile:
    src = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")
print(os.listdir(src['CEC17']))

models = [loadModel(f"{src['CEC17']}/SBX/{m}") for m in os.listdir(f"{src['CEC17']}/SBX")]

fig = CompareModel(
    models=models,
    label=[...] * len(models)
).render(
    shape=(3, 4),
    min_cost=0,
    step=100,
    yscale='log',
    re= True
)