# MUSES: A Curated Benchmark Suite of Event Sequences for Comparing Temporal Point Processes

<img src="figures/logo-light.svg#gh-light-mode-only"
     style="float: right; width: 120px; margin-left: 16px;" />
<img src="figures/logo-dark.svg#gh-dark-mode-only"
     style="float: right; width: 120px; margin-left: 16px;" />


This repository is the official implementation of the benchmark study conducted in [MUSES: A Curated Benchmark Suite of Event Sequences for Comparing Temporal Point Processes](todo). 


 

<h2 style="clear: both;">Requirements</h2>

Setup a `Python 3.11.5` environment and install all dependencies:

```setup
pip install -r requirements.txt
```

## Benchmark Study

To perform the benchmark study, one has to run `./src/main.py --data-id <DATA_ID> --model-id <MODEL_ID>` with various data and model IDs. 
`DATA_ID` indicates the number of the dataset in [MUSES](https://huggingface.co/datasets/ddrg/MUSES) alphabetical ordering.
`MODEL_ID` indicates the number of the model in the `./configs` alphabetical ordering.

We provide a slurm script to easily start the benchmark study on a cluster in `hpc/run_bench.sh`.

Adjust the file to your needs and run:

```benchmark
sbatch hpc/run_bench.sh
```

This will take a while and generate a sqlite database under `./results.db`. 
The final models are saved under `./checkpoints/<dataset_name>/<model_name>/<seed>`.

Note: You might also need to adjust `./hpc/modules.sh` to load the correct modules for your cluster.

## Results

By running our benchmark we achieved the following results across 5 seeds :
# TODO

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |


Note: The exact results may vary depending on how fast the Hardware is, as this allows the optimizer to evaluate more or less configs leading to different outcomes


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 