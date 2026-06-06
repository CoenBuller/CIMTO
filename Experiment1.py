from Experiments.TestAngleOrdering import RunAngleOrdering
from Experiments.TestInitialARM import RunInitialARM
from Experiments.TestInternalARM import RunInternalARM
from Experiments.TestSARTRelaxation import RunRelaxation
from Experiments.TestSmoothing import RunSmoothing

EXPERIMENTS = {"Smoothing": RunSmoothing,
               "Relaxation": RunRelaxation,
               "Internal ARM": RunInternalARM,
               "Initial ARM": RunInitialARM,
               "Projection Ordering": RunAngleOrdering}

for experiment in EXPERIMENTS:
    phantom_cfg = phantomConfig()
    dart_cfg = Config()
    rng = np.random.default_rng(seed=dart_cfg.seed)
    print(f"Starting experiment: {experiment}")
    EXPERIMENTS[experiment]()




