from Experiments.NoisyExperiments.TestInternalARM import RunInternalARM
from Experiments.NoisyExperiments.TestRelaxation import RunRelaxation
from Experiments.NoisyExperiments.TestSmoothing import RunSmoothing
from src.DART.DARTConfig import Config


import numpy as np

EXPERIMENTS = {"Internal ARM": RunInternalARM,
               "Smoothing": RunSmoothing,
               "Relaxation": RunRelaxation,}            

for experiment in EXPERIMENTS:
    dart_cfg = Config()
    print(f"Starting experiment: {experiment}")
    EXPERIMENTS[experiment](dart_config=dart_cfg)




