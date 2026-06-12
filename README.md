# An Empirical Study of Hyperparameter Sensitivity in the Discrete Algebraic Reconstruction Technique

## Requirements

- Python 3.12

Verify your Python version:

```bash
python --version
```

or

```bash
python3 --version
```

The output should show Python 3.12.x.

## Setup
### 1. Create a virtual environment
```bash
python3.12 -m venv .venv
```

### 2. Activate the virtual environment
#### Linux / macOS
```bash
source .venv/bin/activate
```

#### Windows (Command Prompt)
```cmd
.venv\Scripts\activate.bat
```

#### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run & Plot Experiment 1
```bash
python -m Experiment1
```

results should be saved under *./Results/Experiment_1*, there an image can be found. If it is not found there try the following command: 
```bash
python -m Plotting.PlotExperiment1
```

### 5. Run & Plot Experiment 2
```bash
python -m Experiment2
```

results should be saved under *./Results/Experiment_2*, there an image can be found. If it is not found there try the following command: 
```bash
python -m Plotting.PlotExperiment2
```

## Comparing smoothing factors

`CompareSmoothing` runs DART three times on the same phantom under noisy conditions (SNR=20) with smoothing factors σ=0, σ=1, and σ=3, and saves side-by-side reconstruction plots to the working directory.

```bash
python -m Plotting.CompareSmoothing [options]
```

It accepts the same arguments as the DART script (see below), with the following fixed overrides that cannot be changed via arguments:

| Parameter | Fixed value |
|---|---|
| Number of angles | `25` |
| SNR | `20` |
| Smoothing values | `None`, `1`, `3` |

### Example

```bash
python -m CompareSmoothing -type 2 -instance 0 -dart_iters 100 -relaxation 0.9
```

Output files saved to the working directory:
- `CompareSmoothing_phantom_<type>.png` — three reconstructions (σ=0, 1, 3) in the viridis colormap
- `CompareSmoothing_phantom_<type>_gray.png` — Same reconstructions in grayscale colormap

## Running DART directly

DART can also be run as a standalone script, which reconstructs a single phantom and saves a reconstruction image and convergence plot to the working directory.

```bash
python -m DART [options]
```

### Phantom selection

| Argument | Choices | Default | Description |
|---|---|---|---|
| `-type` | `1 2 3 4` | `1` | Phantom type to load from `TestPhantoms/` |
| `-instance` | `0`–`9` | `0` | Which instance of that phantom type to use |
| `-verbal` | `bool` | `False` | Print timing information after the initial reconstruction |

### DART loop

| Argument | Type | Default | Description |
|---|---|---|---|
| `-dart_iters` | `int` | `200` | Total number of DART iterations |
| `-p` | `float` | `0.85` | Probability that a non-edge pixel is kept fixed each iteration. Higher values → fewer free pixels |

### ARM (SART) parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `-init_arm_iters` | `int` | `10` | SART iterations for the initial full-image reconstruction |
| `-arm_iters` | `int` | `3` | SART iterations per inner DART loop (free pixels only) |
| `-relaxation` | `float` | `1.0` | SART relaxation factor λ. Values below 1 give more conservative updates |

### Projection geometry

| Argument | Type | Default | Description |
|---|---|---|---|
| `-n_angles` | `int` | `25` | Number of projection angles |
| `-lower_angle` | `float` | `0` | Start of the angular range (radians) |
| `-upper_angle` | `float` | `π` | End of the angular range (radians) |
| `-angle_ordering` | `str` | `random` | Projection visit order: `random`, `sequential`, or `interleaved` |
| `-gpu` | `bool` | `False` | Use the CUDA-accelerated ASTRA projector |

### Noise

| Argument | Type | Default | Description |
|---|---|---|---|
| `-snr` | `int` | `None` | Target SNR for Poisson noise injection. Omit for a noise-free sinogram |
| `-noise_func` | `str` | `poisson` | Noise model to use (currently only `poisson` is supported) |

### Smoothing

| Argument | Type | Default | Description |
|---|---|---|---|
| `-smoothing` | `float` | `1.0` | Gaussian smoothing σ applied to free pixels after each ARM step. Set to `0` to disable |

### Example

```bash
python -m DART -type 2 -instance 3 -dart_iters 100 -n_angles 40 -init_arm_iters 20 -arm_iters 5 -relaxation 0.9 -angle_ordering interleaved -snr 50 -smoothing 1.5
```

Output files saved to the working directory:
- `DART_reconstruction.png` — side-by-side reconstruction and ground-truth phantom
- `convergence_plot.png` — K-error over iterations