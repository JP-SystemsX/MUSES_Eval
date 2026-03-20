from typer import Typer
from easy_tpp.preprocess import TPPDataLoader
from easy_tpp.runner.base_runner import Runner, logger
from easy_tpp.utils import py_assert
import datasets as ds
import os
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
import yaml 
from pathlib import Path
from ConfigSpace import ConfigurationSpace
from ast import literal_eval
import numpy as np
from copy import deepcopy
from easy_tpp.utils.const import PredOutputIndex
from easy_tpp.utils.metrics import MetricsHelper
from sklearn.metrics import f1_score
from utils import to_builtin, store_complex_dict, string_to_dict
import uuid
import sqlite3
import pandas as pd
import torch


os.chdir(Path(__file__).parent)

os.environ["HF_TOKEN"] = "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" #TODO Remove

# ============================================================= #
# Add support for additional datasources and metrics to EasyTPP #
# ============================================================= #

# Overwrite EasyTPP data loader to also support unofficial datasets
def _build_input_from_json(self, source_dir, split):
    """Load and process data from a JSON file.

    Args:
        source_dir (str): Path to the JSON file or Hugging Face dataset name.
        split (str): Dataset split, e.g., 'train', 'dev', 'test'.

    Returns:
        dict: Dictionary with processed event sequences.
    """
    from datasets import load_dataset
    split_mapped = 'validation' if split == 'dev' else split
    if source_dir.endswith('.json'):
        data = load_dataset('json', data_files={split_mapped: source_dir}, split=split_mapped)
    elif source_dir.startswith('easytpp'):
        data = load_dataset(source_dir, split=split_mapped)
    elif source_dir.count('/') == 2:
        source_dir, ds_name = source_dir.rsplit('/', 1)
        data = load_dataset(source_dir, ds_name, split=split_mapped)
    else:
        raise ValueError("Unsupported source directory format for JSON.")

    py_assert(data['dim_process'][0] == self.num_event_types,
                ValueError, "Inconsistent dim_process in different splits.")

    return {
        'time_seqs': data['time_since_start'],
        'type_seqs': data['type_event'],
        'time_delta_seqs': data['time_since_last_event']
    }

# We replace the evaluate function to return ANY metric instead of only RMSE
def evaluate(self, valid_loader=None, return_all_metrics=False, **kwargs):
    if valid_loader is None:
        valid_loader = self._data_loader.valid_loader()

    logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

    timer = self.timer
    timer.start()
    model_id = self.runner_config.base_config.model_id
    logger.info(f'Start {model_id} evaluation...')

    metric = self._evaluate_model(
        valid_loader,
        **kwargs
    )
    logger.info(f'End {model_id} evaluation! Cost time: {timer.end()}')
    print(f'{model_id} evaluation metric: {metric}')
    if return_all_metrics:
        return metric
    return metric['rmse']  # return a list of scalr for HPO to use


# Monkey Patch
TPPDataLoader._build_input_from_json = _build_input_from_json
Runner.evaluate = evaluate

@MetricsHelper.register(name='f1_macro', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def f1_macro_metric_function(predictions, labels, **kwargs):
    """Compute macro F1 metrics of the type predictions."""
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TypePredIndex]
        label = labels[PredOutputIndex.TypePredIndex]
    else:
        pred = predictions[PredOutputIndex.TypePredIndex][seq_mask]
        label = labels[PredOutputIndex.TypePredIndex][seq_mask]
    return f1_score(label, pred, average='macro')

@MetricsHelper.register(name='f1_micro', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def f1_micro_metric_function(predictions, labels, **kwargs):
    """Compute micro F1 metrics of the type predictions."""
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TypePredIndex]
        label = labels[PredOutputIndex.TypePredIndex]
    else:
        pred = predictions[PredOutputIndex.TypePredIndex][seq_mask]
        label = labels[PredOutputIndex.TypePredIndex][seq_mask]
    return f1_score(label, pred, average='micro')



app = Typer(pretty_exceptions_enable=False)

@app.command()
def main(data_id: int = 0, model_id: int = 0, trial_count: int = 2):
    search_id = uuid.uuid4()

    ds_names = sorted(ds.get_dataset_config_names("ddrg/NEDTBench"))
    ds_name = ds_names[data_id]

    meta_data = ds.load_dataset_builder("ddrg/NEDTBench", ds_name)
    class_count = len(meta_data.info.features["type_event"].feature.names)

    model_configs = Path("../configs/model_configs/").glob("*.yml")
    model_configs = sorted(model_configs, key=lambda x: x.stem)
    model_config_file = model_configs[model_id]
    model_name = model_config_file.stem

    # save config to yaml
    config_address = Path(f'../tmp/{model_name}-{ds_name}-{search_id}.yaml')
    config_address.parent.mkdir(exist_ok=True, parents=True)
    config_ = {
        "pipeline_config_id": "runner_config",
        "data": {
            ds_name: {
                "data_format": "json",
                "train_dir": f"ddrg/NEDTBench/{ds_name}",
                "valid_dir": f"ddrg/NEDTBench/{ds_name}",
                "test_dir": f"ddrg/NEDTBench/{ds_name}",
                "data_specs": {
                    "num_event_types": class_count,
                    "pad_token_id": class_count,
                    "padding_side": "right"
                }
            }
        },
    }


    # Simple Random Search
    # Load Search Space from yaml
    search_space = yaml.safe_load(open(model_config_file))
    for key in search_space.keys():
        for sub_key in search_space[key].keys():
            if isinstance(search_space[key][sub_key], str) and search_space[key][sub_key].startswith("(") and search_space[key][sub_key].endswith(")"):
                search_space[key][sub_key] = literal_eval(search_space[key][sub_key])
    cs_trainer = ConfigurationSpace(search_space["trainer_config"])
    cs_model = ConfigurationSpace(search_space["model_config"])

    for trainer_cfg, model_cfg in zip(list(cs_trainer.sample_configuration(size=trial_count)), list(cs_model.sample_configuration(size=trial_count))):
        print("Trial with configs: ", trainer_cfg, model_cfg)
        config = deepcopy(config_)
        trainer_cfg = dict(trainer_cfg)
        trainer_cfg = {k: to_builtin(v) for k, v in trainer_cfg.items()}
        model_cfg = dict(model_cfg)
        model_cfg = {k: to_builtin(v) for k, v in model_cfg.items()}
        trainer_cfg["metrics"] = ['acc', 'rmse', 'f1_macro', 'f1_micro'] 
        config["train"] = {
            "base_config": {
                "stage": "train",
                "backend": "torch",
                "dataset_id": ds_name, 
                "runner_id": "std_tpp",
                "model_id": model_name, 
                "base_dir": "../checkpoints/",
            },
            "trainer_config": trainer_cfg,
            "model_config": model_cfg,
        }
        # save config to yaml
        with open(config_address, 'w') as f:
            yaml.safe_dump(config, f)

        runner_config = Config.build_from_yaml_file(str(config_address), experiment_id="train")

        model_runner = Runner.build_from_config(runner_config)

        model_runner.run()
        results = model_runner.evaluate(return_all_metrics=True)

        # Log results to DB
        config["search_id"] = str(search_id)
        config.update(results)
        store_complex_dict(config, database_path="../results.db", table_name="trials")
        print("Hello EasyTPP!")

    # Get best config from db
    conn = sqlite3.connect("../results.db")
    df = pd.read_sql_query("SELECT * FROM trials WHERE search_id = ?", conn, params=[str(search_id)])
    min_rmse = df["rmse"].min()
    max_rmse = df["rmse"].max()
    df["score"] = df["f1_macro"] + df["f1_micro"] + df["acc"] - 3 * (df["rmse"] - min_rmse) / (max_rmse - min_rmse)  
    best_row = df.loc[df["score"].idxmax()]
    print("Best Config: ", best_row)

    # Retrain best config
    best_config = {
        "pipeline_config_id": best_row["pipeline_config_id"],
        "data": string_to_dict(best_row["data"]),
        "train": string_to_dict(best_row["train"]),
    }
    # best_config["train"]["trainer_config"]["metrics"] = []


    for seed in range(5): 
        print(f"Retraining with best config, seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Also adjust seed of the trainer
        best_config["train"]["trainer_config"]["seed"] = seed
        with open(config_address, 'w') as f:
            yaml.safe_dump(best_config, f)
        
        runner_config = Config.build_from_yaml_file(str(config_address), experiment_id="train")

        model_runner = Runner.build_from_config(runner_config)

        model_runner.run()
        valid_results = model_runner.evaluate(return_all_metrics=True)
        test_results = model_runner.evaluate(model_runner._data_loader.test_loader(), return_all_metrics=True) # TODO I think making a prediction after the sequence has ended, leads to worthless forecasts (PAD tokens) --> Eval instead.
        train_results = model_runner.evaluate(model_runner._data_loader.train_loader(), return_all_metrics=True) # To check for overfitting

        results = deepcopy(best_config)
        results["seed"] = seed
        for split, r in[("valid", valid_results), ("test", test_results), ("train", train_results)]:
            for metric_name, metric_value in r.items():
                results[f"{split}_{metric_name}"] = metric_value

        store_complex_dict(results, database_path="../results.db", table_name="final_results")

    # TODO Load MetaData
    # TODO Load Data
    # TODO Load Model
    # TODO Optimize Model
    # TODO Evaluate Model
    # TODO Save Predictions
    pass


if __name__ == "__main__":
    app()