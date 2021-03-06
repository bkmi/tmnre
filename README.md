# Truncated Marginal Neural Ratio Estimation

This repository is the official implementation of the experiments from Truncated Marginal Neural Ratio Estimation, published in NeurIPS 2021. Read the preprint on [arxiv](https://arxiv.org/abs/2107.01214).
The underlying algorithm implementation is called [swyft](https://github.com/undark-lab/swyft) and we perform experiments with [sbi](https://github.com/mackelab/sbi) and a forked version of [sbibm](https://github.com/bkmi/sbibm). 

The results data, explained below, are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5592427.svg)](https://doi.org/10.5281/zenodo.5592427). These data are rather large and generally unnecessary for using the repsoitory. They represent the output of the TMNRE method on problems presented within the paper.

## Setup

### Submodules

First, pull the submodules to use the repository.

```setup
git submodule init
git submodule update
```

### Requirements & Installation

Next, install the requirements using conda. Run the following command. Note, it assumes that you can install `cudatoolkit=11.1`. If not, change the `environment.yml` file accordingly. [See pytorch's website](https://pytorch.org/get-started/locally/).

```setup
conda env create --prefix .tmnre_env --file environment.yml
conda activate ./.tmnre_env
```

The above command will also install the `tmnre` package from this repository.

### Accompanying Data - Zenodo

There is a set of accompanying data (>27GB) including simulations, results, and more. *You do not need the accompanying data to use most of the repository*. The data is available on [Zenodo](https://doi.org/10.5281/zenodo.5592427). Each of the files were compressed using gzip. The manifest reads:

```
inference_sbibm_raw.tar.gz
marginalized_sbibm_raw.tar.gz
TTTEEElm2500.zarr.tar.gz
```

- `inference_sbibm_raw.tar.gz` corresponds to the data generated when applying `tmnre` to `sbibm`. It can be extracted to `inference_sbibm/raw`.  
- `marginalized_sbibm_raw.tar.gz` corresponds to the metrics calculated from marginalizing the posterior samples of other `sbi` algorithms in `sbibm`. It can be extracted to `marginalized_sbibm/raw`.  
- Finally, `TTTEEElm2500.zarr.tar.gz` contains simulations for the physics example. It **must** be extracted to `physics/TTTEEElm2500.zarr` for the physics notebook to function correctly.

## sbibm results

### Marginalizing sbibm

Performing the marginalization of `sbibm` requires the [very large raw dataset from sbibm](https://github.com/mackelab/benchmarking_sbi_runs).
If you would like to install it so you can perform the marginalization, you must first install [Git LFS](https://git-lfs.github.com/) then execute the following commands.

```setup
cd remote
git submodule add git@github.com:mackelab/benchmarking_sbi_runs.git
```

We use [hydra](https://hydra.cc/) to perform the marginalization and calculation of the c2st-ddm performance metric on the existing `sbibm` posterior data. Assuming you installed the raw `sbibm` dataset (see above), you can run the following command to calculate the c2st for every 1-dim and 2-dim marginal from the slcp task estimated by the nre algorithm with the following command:

```bash
python marginalize_sbibm/calc_marginalized_c2st.py raw_results_root=remote/benchmarking_sbi_runs/runs/ algorithm=nre task=slcp
```

If you were interested in several algorithms...

```bash
python marginalize_sbibm/calc_marginalized_c2st.py raw_results_root=remote/benchmarking_sbi_runs/runs/ algorithm=nre,npe,snre,snpe task=slcp -m
```

Note that the `-m` flag is required for hydra's multirun. Calculating the c2st for every marginal, observation, algorithm, and task is expensive. It requires training over 100,000 classifiers. We would recommend consulting hydra's documentation to easily use slurm or another scheduler to accomplish the task with multirun.

Both of the above commands produce output according to hydra's default behavior.

Finally, once you've calculated the metrics of interest, we can summarize the results with another command:

```bash
python marginalize_sbibm/summarize_marginalized_c2st.py ~/path/to/marginals marginal-sbibm-results.csv
```

We have provided a results file already calculated in `marginalize_sbibm/marginal-c2st-summary.csv`. It was computed using the raw data from our marginalization runs. Those files are stored in `marginalized_sbibm_raw.tar.gz` on Zenodo.

### tmnre on sbibm

Training and evaluating `tmnre` on a selection of `sbibm` tasks can be done using three programs in the `inference_sbibm` folder, namely `run.py`, `summarize.py`, and `metrics.py`. For training, preparation for plotting, and evaluation respectively. `run.py` implicitly calls `metrics.py` if you setup the hydra configuration correctly. There are many configuration options, we encourage the user to explore them by running `python inference_sbibm/run.py -h`. The following is a simple example run which trains, then generates the performance metric information for every 1-dim and 2-dim marginal.

```bash
python inference_sbibm/run.py task=two_moons task.num_simulations=10_000 task.num_observation=2 hardware=cuda0 analysis=metrics_2d_marginals 
```

Once you have done training and evaluation you can make a summary dataframe and a plot. First summarize the data using the following command.

```bash
python inference_sbibm/summarize.py ~path/to/data tmnre-sbibm-results.csv
```

We have provided two files already computed in `inference_sbibm/reports/swyft_uniform_2d_results_budget.csv` and `inference_sbibm/reports/swyft_non_uniform_2d_results_budget.csv`. They were computed using the raw data from our experimental runs. The corresponding raw data files are stored in `inference_sbibm_raw.tar.gz` on Zenodo.

### Generating the comparison plot

The jupyter notebook `inference_sbibm/compare.ipynb` is designed to use the plotting functions built into `sbibm` to create the plot that we reported in our paper. It requires three summary dataframes, `marginalize_sbibm/marginal-c2st-summary.csv`, `inference_sbibm/reports/swyft_uniform_2d_results_budget.csv`, and `inference_sbibm/reports/swyft_non_uniform_2d_results_budget.csv`. They have already been provided for you and will be referenced if you simply run all the cells of the notebook.

## Torus

### Ground Truth

The ground truth for this task was generated using rejection sampling. The task itself is defined within the `sbibm` framework. The ten observations and corresponding ground truth posterior sample data is presented, along with the code to generate it, in `remote/sbibm/torus`. All observations are generated with the same underlying parameter but different noise realizations. The first observation is generated without noise.

### Epsilon Hyperparameter Sweep

The torus task is defined within our custom version of `sbibm`, a submodule in this package. Therefore, we can train on that task simply by calling a script which applies `tmnre` to the problem. We also applied hydra to this experiment. You can specify the hardware to train on using options `cpu` and `cuda0`. Epsilon is input as the exponent of a power of 10. The default `-6` implies `epsilon=10 ** -6`.

```bash
python torus/epsilon/torus.py hardware=cuda0 epsilon=-6
```

Once you have run the experiment multiple times, we create a summary dataframe like so:

```bash
python torus/epsilon/summarize.py ~/path/to/torus_sweep epsilon-results.csv
```

### Comparing tmnre to mnre

The training of these methods is implemented in a jupyter notebook `gen-torus.ipynb` with diagnostic plots. When the `SAVE` flag is turned on, the notebook trains and generates a number of data pickles. The ones reported in the paper are included within this repo.

### Other SBI Methods

The other sbi methods were trained using the script `torus/metrics/sbi-torus.py`. The corresponding plots were generated using `torus/metrics/sbi-plot-torus.ipynb`.

## Eggbox

### Ground Truth

The ground truth for this task was generated using `D` 1-dimensional independent draws from a numerical inverse transform sampling method. The task itself is defined within the `sbibm` framework. we only present a single observation as reported in the paper. The code to generate ground truth data of any dimensionality can be found in `remote/sbibm/eggbox` along with 3, 5, 10, 25 dimensional ground truth posterior samples.

### Comparing mnre to nre, snre, smnre

Training `mnre`, `nre`, and `snre` are all implemented within a jupyter notebook, along with diagnostic plots, in `eggbox/gen-eggbox.ipynb`. If the `SAVE` flag is turned on, the notebook trains and generates data pickles which contain c2st results, reference posterior samples, and estimated posterior samples. The data pickles we reported are already included in this repo. Finally, `msnre` is trained in another notebook `eggbox/smnre-eggbox-few.ipynb` where it produces posterior samples in another pickle, also included. Finally, the reported plots are created in the `eggbox/plot-eggbox.ipynb` notebook.

### Other SBI Methods

The other sbi methods were trained using the script `eggbox/sbi-eggbox.py`. The corresponding plots were generated using `eggbox/metrics/sbi-plot-eggbox.ipynb`.

### Rotated Eggbox

The details of the rotated eggbox are all within a folder called `eggbox/rotated`.

## Physics example

The cosmology simulator uses the code [CLASS](https://github.com/lesgourg/class_public/), but we have saved the zarr simulation "datastore" so the simulator doesn't need to be called. This file can be found in `TTTEEElm2500.zarr.tar.gz` on Zenodo. Processed data in the form of plots and notebooks are available within the `physics` folder.)  Simply set the `LOAD` flag to `True` and the notebook `CMB_TTTEEE.ipynb` will run. The observation `obs` is also shipped with the repository -- this is needed to define the stochastic part of the simulator. The plots are made using [getdist](https://github.com/cmbant/getdist).

# Results

## sbibm

![two_moons](inference_sbibm/figures/two_moons.png)
![gaussian_linear_uniform](inference_sbibm/figures/gaussian_linear_uniform.png)
![slcp](inference_sbibm/figures/slcp.png)
![slcp_distractors](inference_sbibm/figures/slcp_distractors.png)
![gaussian_mixture](inference_sbibm/figures/gaussian_mixture.png)

## Torus

A few posteriors from the torus problem. Ground truth, `tmnre`, and mnre.  
<img src="torus/metrics/figures/torus-ref-corner.png" width="150" height="150">
<img src="torus/metrics/figures/torus-tmnre-corner.png" width="150" height="150">
<img src="torus/metrics/figures/torus-mnre-corner.png" width="150" height="150">

A comparison of mnre to `tmnre` with several metrics.  
![metrics](torus/metrics/figures/torus-metrics.png)

Results of our hyperparameter scan.  
![epsilon](torus/epsilon/figures/torus-epsilon.png)

## Eggbox

A few posteriors from the eggbox problem. Ground truth, mnre, nre, and smnre.  
<img src="eggbox/figures/eggbox-ref-corner.png" width="120" height="120">
<img src="eggbox/figures/eggbox-mnre-corner.png" width="120" height="120">
<img src="eggbox/figures/eggbox-nre-corner.png" width="120" height="120">
<img src="eggbox/figures/eggbox-snre-corner.png" width="120" height="120">
<img src="eggbox/figures/eggbox-msnre-corner.png" width="120" height="120">


## Stochasticity

The simulation and training is done in parallel. That means that even with a fixed seed, it is usually impossible to return exactly the same results. Data is therefore provided to reproduce our plots.

## Contributing

Please apply the isort and black formatters to any submitted code.
