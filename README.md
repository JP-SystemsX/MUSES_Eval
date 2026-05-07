# MUSES: A Curated Benchmark Suite of Event Sequences for Comparing Temporal Point Processes

<img src="figures/logo-light.svg#gh-light-mode-only"
     align="right"
     width="300" />
<img src="figures/logo-dark.svg#gh-dark-mode-only"
     align="right"
     width="300" />
This repository is the official implementation of the benchmark study conducted in [MUSES: A Curated Benchmark Suite of Event Sequences for Comparing Temporal Point Processes](todo). 

<br clear="both">
 
 ## Requirements

Setup a `Python 3.11.5` environment and install all dependencies:

```setup
pip install -r requirements.txt
```

## Benchmark Study

To perform the benchmark study, one has to run `./src/main.py --data-id <DATA_ID> --model-id <MODEL_ID>` with various data and model IDs. 
`DATA_ID` indicates the number of the dataset in [MUSES](https://huggingface.co/datasets/ddrg/MUSES) alphabetical ordering.
`MODEL_ID` indicates the number of the model in the `./configs` folder in alphabetical ordering.

We provide a slurm script to easily start the benchmark study on a cluster in `hpc/run_bench.sh`.

Adjust the file to your needs and run:

```benchmark
sbatch hpc/run_bench.sh
```

This will take a while and generate a sqlite database under `./results.db`. 
The final models are saved under `./checkpoints/<dataset_name>/<model_name>/<seed>`.

Note: You might also need to adjust `./hpc/modules.sh` to load the correct modules for your cluster.

## Evaluation

We provide a notebook to evaluate (or rather visualize) the results of the benchmark study in `./analysis/benchmark.ipynb`.
This allows to convert the raw results into latex tables, calculate normalized errors, create Critical Difference Diagrams, and more.


## Results

By running our benchmark we achieved the following normalized results across 5 seeds and 18 datasets:


| Metric | NHP | RMTPP | S2P2 | SAHP | THP |
|---|---:|---:|---:|---:|---:|
| Acc | **0.53** | 0.80 | 0.85 | 0.67 | 0.60 |
| RMSE | 0.75 | 0.60 | 0.90 | **0.59** | 0.72 |
| Compute | 0.95 | **0.17** | 0.93 | 0.75 | 0.88 |
| Objective | **0.40** | 0.88 | 0.78 | 0.81 | 0.69 |


Note: The exact results may vary depending on how fast the Hardware is, as this allows the optimizer to evaluate more or less configs leading to different outcomes



