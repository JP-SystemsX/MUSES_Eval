import datetime
import sys
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
from easy_tpp.config_factory.model_config import ModelConfig
from sklearn.metrics import f1_score
from utils import to_builtin, store_complex_dict, string_to_dict, set_seed
import uuid
import sqlite3
import pandas as pd
from tqdm.auto import tqdm
from time import time
from datasets import load_dataset
import gc
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


def get_item_override(self, key):
    """Some models access config via get_item instead of attribute access. This workaround allows both."""
    if key == "dropout":
        return self.dropout_rate # dropout is under different names in different models
    return getattr(self, key)
ModelConfig.__getitem__ = get_item_override



app = Typer(pretty_exceptions_enable=False)

@app.command()
def main(data_id: int = 0, model_id: int = 0, trial_count: int = 2):
    search_id = uuid.uuid4()
    set_seed(42)

    ds_names = sorted(ds.get_dataset_config_names("ddrg/NEDTBench"))
    ds_name = ds_names[data_id]

    meta_data = ds.load_dataset_builder("ddrg/NEDTBench", ds_name)
    class_count = len(meta_data.info.features["type_event"].feature.names)

    model_configs = Path("../configs/model_configs/").glob("*.yml")
    model_configs = sorted(model_configs, key=lambda x: x.stem)
    model_config_file = model_configs[model_id]
    model_name = model_config_file.stem

    # Skip if already done
    done_file = Path("../tmp/done.txt")
    if done_file.exists():
        with open(done_file, "r", encoding="utf-8") as f:
            done_lines = f.read().splitlines()
            if f"{model_name}|{ds_name}" in done_lines:
                print(f"Skipping {model_name} on {ds_name} as it's already done.")
                return

    # save config to yaml
    config_address = Path(f'../tmp/{model_name}-{ds_name}-{search_id}.yaml')
    config_address.parent.mkdir(exist_ok=True, parents=True)
    config_ = {
        "pipeline_config_id": "runner_config",
        "data":   {
            ds_name: {
                "data_format": "json",
                "train_dir": f"ddrg/NEDTBench/{ds_name}",
                "valid_dir": f"ddrg/NEDTBench/{ds_name}",
                "test_dir": f"ddrg/NEDTBench/{ds_name}",
                "data_specs": {
                    "num_event_types": class_count,
                    "pad_token_id": class_count,
                    "padding_side": "left", 
                }
            }
        },
    }

    # Load Dataset to get more detailed meta data
    dataset = load_dataset("ddrg/NEDTBench", ds_name, split="train")
    # Cap maxlength to 1000 (to avoid OOM)
    if max(dataset["seq_len"]) > 1000:
        config_["data"]["data_specs"]["max_length"] = 1000
        config_["data"]["data_specs"]["padding_strategy"] = "max_length"
        config_["data"]["data_specs"]["truncation_strategy"] = "longest_first"


    # Simple Random Search
    # Load Search Space from yaml
    model_search_space = yaml.safe_load(open(model_config_file))
    trainer_search_space = yaml.safe_load(open("../configs/trainer_search_space.yml"))
    for search_space in [model_search_space, trainer_search_space]:
        for key in search_space.keys():
            if isinstance(search_space[key], str) and search_space[key].startswith("(") and search_space[key].endswith(")"):
                search_space[key] = literal_eval(search_space[key])

    cs_trainer = ConfigurationSpace(trainer_search_space)
    cs_model = ConfigurationSpace(model_search_space)

    search_start_time = time()
    for trainer_cfg, model_cfg in tqdm(zip(list(cs_trainer.sample_configuration(size=trial_count)), list(cs_model.sample_configuration(size=trial_count))), total=trial_count):
        print("Trial with configs: ", trainer_cfg, model_cfg)
        if time() - search_start_time > 48 * 60 * 60:  # Stop search after 48 hours
            print("Stopping search after 48 hours.")
            break

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        config = deepcopy(config_)
        trainer_cfg = dict(trainer_cfg)
        trainer_cfg = {k: to_builtin(v) for k, v in trainer_cfg.items()}
        model_cfg = dict(model_cfg)
        model_cfg = {k: to_builtin(v) for k, v in model_cfg.items()}
        trainer_cfg["metrics"] = ['acc', 'rmse', 'f1_macro'] 
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

        try:
            start_time = time()
            model_runner = Runner.build_from_config(runner_config)

            model_runner.run()
            results = model_runner.evaluate(return_all_metrics=True)

            # Log results to DB
            config["search_id"] = str(search_id)
            config.update(results)
            config["trial_time"] = time() - start_time
            config["timestamp"] = datetime.datetime.now().isoformat()
            config["dataset"] = ds_name
            config["model"] = model_name
            store_complex_dict(config, database_path="../results.db", table_name="trials")
        except Exception as e:
            print(f"Error in trial with configs: {trainer_cfg}, {model_cfg}. Error: {e}", flush=True, file=sys.stderr)
            continue

    # Get best config from db
    conn = sqlite3.connect("../results.db")
    df = pd.read_sql_query("SELECT * FROM trials WHERE search_id = ?", conn, params=[str(search_id)])
    min_rmse = df["rmse"].min()
    max_rmse = df["rmse"].max()
    df["score"] = df["f1_macro"] + df["acc"] - 2 * ((df["rmse"] - min_rmse) / (max_rmse - min_rmse))  
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
        set_seed(seed)

        # Also adjust seed of the trainer
        best_config["train"]["trainer_config"]["seed"] = seed
        with open(config_address, 'w') as f:
            yaml.safe_dump(best_config, f)
        
        runner_config = Config.build_from_yaml_file(str(config_address), experiment_id="train")

        model_runner = Runner.build_from_config(runner_config)

        model_runner.run()
        valid_results = model_runner.evaluate(return_all_metrics=True)
        test_results = model_runner.evaluate(model_runner._data_loader.test_loader(), return_all_metrics=True) 
        train_results = model_runner.evaluate(model_runner._data_loader.train_loader(), return_all_metrics=True) # To check for overfitting

        results = deepcopy(best_config)
        results["seed"] = seed
        results["dataset"] = ds_name
        results["model"] = model_name
        for split, r in[("valid", valid_results), ("test", test_results), ("train", train_results)]:
            for metric_name, metric_value in r.items():
                results[f"{split}_{metric_name}"] = metric_value

        store_complex_dict(results, database_path="../results.db", table_name="final_results")
    
    # Mark as finished
    with open(done_file, "a", encoding="utf-8") as f:
        f.write(f"{model_name}|{ds_name}\n")



if __name__ == "__main__":
    app()