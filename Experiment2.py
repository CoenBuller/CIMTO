from Experiments.NoisyExperiments.TestInternalARM import RunInternalARM
from Experiments.NoisyExperiments.TestRelaxation import RunRelaxation
from Experiments.NoisyExperiments.TestSmoothing import RunSmoothing
from src.DART.DARTConfig import Config

from Plotting.PlotExperiment2 import plot_all_experiments_combined

"""Executes experiment 1 and plot the results. It stores the final figure under ./Results/Experiment_2"""

EXPERIMENTS = {"Internal ARM": RunInternalARM,
               "Smoothing": RunSmoothing,
               "Relaxation": RunRelaxation,}            

for experiment in EXPERIMENTS:
    dart_cfg = Config()
    print(f"Starting experiment: {experiment}")
    EXPERIMENTS[experiment](dart_config=dart_cfg)

plot_all_experiments_combined()




