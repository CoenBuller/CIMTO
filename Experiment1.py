from Experiments.TestAngleOrdering import RunAngleOrdering
from Experiments.TestInitialARM import RunInitialARM
from Experiments.TestInternalARM import RunInternalARM
from Experiments.TestSARTRelaxation import RunRelaxation
from Experiments.TestSmoothing import RunSmoothing
from src.DART.DARTConfig import Config

from Plotting.PlotExperiment1 import plot_all_experiments_combined

"""Executes experiment 1 and plot the results. It stores the final figure under ./Results/Experiment_1. Parameters are varied are shown in individual experiment files."""

EXPERIMENTS = {"Projection Ordering": RunAngleOrdering,
               "Smoothing": RunSmoothing,
               "Relaxation": RunRelaxation,
               "Internal ARM": RunInternalARM,
               "Initial ARM": RunInitialARM,}
               

for experiment in EXPERIMENTS:
    dart_cfg = Config()
    print(f"Starting experiment: {experiment}")
    EXPERIMENTS[experiment](dart_config=dart_cfg)

plot_all_experiments_combined()



