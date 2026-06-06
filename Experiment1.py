from Experiments.TestAngleOrdering import RunAngleOrdering
from Experiments.TestInitialARM import RunInitialARM
from Experiments.TestInternalARM import RunInternalARM
from Experiments.TestSARTRelaxation import RunRelaxation
from Experiments.TestSmoothing import RunSmoothing
from src.DART.DARTConfig import Config

import numpy as np

EXPERIMENTS = {"Smoothing": RunSmoothing,
               "Relaxation": RunRelaxation,
               "Internal ARM": RunInternalARM,
               "Initial ARM": RunInitialARM,
               "Projection Ordering": RunAngleOrdering}

for experiment in EXPERIMENTS:
    dart_cfg = Config()
    print(f"Starting experiment: {experiment}")
    EXPERIMENTS[experiment](dart_config=dart_cfg)




