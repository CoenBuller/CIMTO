from Experiments.NoisyExperiments.TestInternalARM import RunInternalARM
from Experiments.NoisyExperiments.TestRelaxation import RunRelaxation
from Experiments.NoisyExperiments.TestSmoothing import RunSmoothing
from src.DART.DARTConfig import Config


import numpy as np

EXPERIMENTS = {"Smoothing": RunSmoothing,
               "Relaxation": RunRelaxation,
               "Internal ARM": RunInternalARM,}               

for experiment in EXPERIMENTS:
    dart_cfg = Config()
    print(f"Starting experiment: {experiment}")
    EXPERIMENTS[experiment](dart_config=dart_cfg)




