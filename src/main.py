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
from easy_tpp.utils import RunnerPhase
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.torch_wrapper import TorchModelWrapper
from sklearn.metrics import f1_score, r2_score
from utils import store_complex_dict, string_to_dict, set_seed, trial
import uuid
import sqlite3
import pandas as pd
from tqdm.auto import tqdm
from time import time
from datasets import load_dataset
import torch
from dehb import DEHB
from functools import partial
from gluonts.evaluation.metrics import smape
from constants import expensive_thinning, cheap_thinning


os.chdir(Path(__file__).parent)


# ============================================================= #
# Add support for additional datasources and metrics to EasyTPP #
# ============================================================= #

# Overwrite EasyTPP data loader to also support unofficial datasets + allow subsampling for large datasets to speed up HPO
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
    elif source_dir.count('/') == 3: # Load and Cap Hugging Face dataset for HPO to speed it up, format should be "ddrg/MUSES/ds_name/ds_length_cap"
        source_dir, ds_name, ds_length_cap = source_dir.rsplit('/', 2)
        ds_length_cap = int(ds_length_cap)
        data = load_dataset(source_dir, ds_name, split=split_mapped)
        if split_mapped in ['train', "validation"] and len(data) > ds_length_cap:  # Cap training set to specified number of samples for HPO to speed it up
            data = data.shuffle(seed=42).select(range(ds_length_cap))
    else:
        raise ValueError("Unsupported source directory format for JSON.")

    py_assert(data['dim_process'][0] == self.num_event_types,
                ValueError, "Inconsistent dim_process in different splits.")

    print(f"Loaded {len(data)} sequences from {source_dir} for split {split}.")
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


def run_batch(self, batch, phase):
    """Run one batch.

    Args:
        batch (EasyTPP.BatchEncoding): preprocessed batch data that go into the model.
        phase (RunnerPhase): a const that defines the stage of model runner.

    Returns:
        tuple: for training and validation we return loss, prediction and labels;
        for prediction we return prediction.
    """

    batch = batch.to(self.device).values()
    if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
        # set mode to train
        is_training = (phase == RunnerPhase.TRAIN)
        self.model.train(is_training)

        # FullyRNN needs grad event in validation stage
        grad_flag = is_training if not self.model_id == 'FullyNN' else True
        # run model
        with torch.set_grad_enabled(grad_flag):
            loss, num_event = self.model.loglike_loss(batch)

        # Assume we dont do prediction on train set
        pred_dtime, pred_type, label_dtime, label_type, mask = None, None, None, None, None

        # update grad
        if is_training:
            if num_event > 0: # ! Sometimes num_event can be 0 due to padding, we should skip those batches to division by zero
                self.opt.zero_grad()
                (loss / num_event).backward()
                self.opt.step()
        else:  # by default we do not do evaluation on train set which may take a long time
            if self.model.event_sampler:
                self.model.eval()
                with torch.no_grad():
                    if batch[1] is not None and batch[2] is not None:
                        label_dtime, label_type = batch[1][:, 1:].cpu().numpy(), batch[2][:, 1:].cpu().numpy()
                    if batch[3] is not None:
                        mask = batch[3][:, 1:].cpu().numpy()
                    pred_dtime, pred_type = self.model.predict_one_step_at_every_event(batch=batch)
                    pred_dtime = pred_dtime.detach().cpu().numpy()
                    pred_type = pred_type.detach().cpu().numpy()
        return loss.item(), num_event, (pred_dtime, pred_type), (label_dtime, label_type), (mask,)
    else:
        pred_dtime, pred_type, label_dtime, label_type = self.model.predict_multi_step_since_last_event(batch=batch)
        pred_dtime = pred_dtime.detach().cpu().numpy()
        pred_type = pred_type.detach().cpu().numpy()
        label_dtime = label_dtime.detach().cpu().numpy()
        label_type = label_type.detach().cpu().numpy()
        return (pred_dtime, pred_type), (label_dtime, label_type)


# Monkey Patch
TPPDataLoader._build_input_from_json = _build_input_from_json
TorchModelWrapper.run_batch = run_batch
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


@MetricsHelper.register(name='r2', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def r2_metric_function(predictions, labels, **kwargs):
    """Compute R2 metrics of the time predictions."""
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TimePredIndex]
        label = labels[PredOutputIndex.TimePredIndex]
    else:
        pred = predictions[PredOutputIndex.TimePredIndex][seq_mask]
        label = labels[PredOutputIndex.TimePredIndex][seq_mask]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return r2_score(label, pred)


@MetricsHelper.register(name='smape', direction=MetricsHelper.MINIMIZE, overwrite=False)
def smape_metric_function(predictions, labels, **kwargs):
    """Compute SMAPE metrics of the time predictions."""
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TimePredIndex]
        label = labels[PredOutputIndex.TimePredIndex]
    else:
        pred = predictions[PredOutputIndex.TimePredIndex][seq_mask]
        label = labels[PredOutputIndex.TimePredIndex][seq_mask]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return smape(label, pred) 



app = Typer(pretty_exceptions_enable=False)

@app.command()
def main(data_id: int = 2, model_id: int = 2, trial_count: int = 2, seed: int = 42):
    SEARCH_DURATION = 24 * 3600 # TODO 48 * 3600  # 48 hours in seconds
    search_id = uuid.uuid4()
    set_seed(seed)

    ds_names = sorted(ds.get_dataset_config_names("ddrg/MUSES"))
    ds_name = ds_names[data_id]

    meta_data = ds.load_dataset_builder("ddrg/MUSES", ds_name)
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
    train_config_address = Path(f'../tmp/best/{model_name}-{ds_name}-{search_id}-train.yaml')
    eval_config_address = Path(f'../tmp/best/{model_name}-{ds_name}-{search_id}-eval.yaml')
    train_config_address.parent.mkdir(exist_ok=True, parents=True)
    config_ = {
        "pipeline_config_id": "runner_config",
        "data":   {
            ds_name: {
                "data_format": "json",
                "train_dir": f"ddrg/MUSES/{ds_name}",
                "valid_dir": f"ddrg/MUSES/{ds_name}",
                "test_dir": f"ddrg/MUSES/{ds_name}",
                "data_specs": {
                    "num_event_types": class_count,
                    "pad_token_id": class_count,
                    "padding_side": "right", 
                }
            }
        },
    }

    # Load Dataset to get more detailed meta data
    dataset = load_dataset("ddrg/MUSES", ds_name, split="train")
    # Cap maxlength to 1000 (to avoid OOM)
    if max(dataset["seq_len"]) > 1000:
        config_["data"][ds_name]["data_specs"]["max_len"] = 1000
        config_["data"][ds_name]["data_specs"]["padding_strategy"] = "max_length"
        config_["data"][ds_name]["data_specs"]["truncation_strategy"] = "longest_first"


    # Simple Random Search
    # Load Search Space from yaml
    model_search_space = yaml.safe_load(open(model_config_file))
    trainer_search_space = yaml.safe_load(open("../configs/trainer_search_space.yml"))
    for search_space in [model_search_space, trainer_search_space]:
        for key in search_space.keys():
            if isinstance(search_space[key], str) and search_space[key].startswith("(") and search_space[key].endswith(")"):
                search_space[key] = literal_eval(search_space[key])

    if "model_specs" in model_search_space:
        model_specs_search_space = model_search_space.pop("model_specs")
    else:
        model_specs_search_space = {}

    # Combine three search spaces to flatten them (Easy TPP configs are nested ConfigSpace works best with flat search spaces)
    trainer_search_space = {f"train.{key}": value for key, value in trainer_search_space.items()}
    model_search_space = {f"model.{key}": value for key, value in model_search_space.items()}
    model_specs_search_space = {f"model_specs.{key}": value for key, value in model_specs_search_space.items()}
    search_space = {**trainer_search_space, **model_search_space, **model_specs_search_space}
    cs = ConfigurationSpace(search_space)
    cs["train.learning_rate"].log = True


    # ============= #
    # Random Search #
    # ============= #
    search_start_time = time()

    dim = len(list(cs.values()))
    optimizer = DEHB(
        cs=cs,
        f=partial(
            trial,
            model_name=model_name,
            ds_name=ds_name,
            search_id=search_id,
            config_=config_,
            seed=seed,
            search_start_time=search_start_time,
            max_time_budget=SEARCH_DURATION + 60,  # Abort Trials when budget used up (if DEHB does not)
        ),
        dimensions=dim,
        min_fidelity=4,  # Low Fidelity
        max_fidelity=64, # High Fidelity
        eta=4,  
        n_workers=1, # TODO
        output_path=f"../hpc/dehb_logs/{model_name}_{ds_name}.log",
        seed=seed,
    )

    # Run optimization for 1 bracket. Output files will be saved to ./logs
    optimizer.run(fevals=trial_count, total_cost=SEARCH_DURATION)  
    search_duration = time() - search_start_time
    print(f"Search finished in {search_duration/3600:.2f} hours.")


    # Get best config from db
    conn = sqlite3.connect("../results.db")
    df = pd.read_sql_query("""
    SELECT t.*
    FROM trials t
    JOIN (
        SELECT search_id, MAX(fidelity) AS max_fidelity
        FROM trials
        WHERE search_id = ?
    ) m
    ON t.search_id = m.search_id
    AND t.fidelity = m.max_fidelity;
    """, conn, params=[str(search_id)])
    best_row = df.loc[df["loglike"].idxmax()]
    print("Best Config: ", best_row)

    # Retrain best config
    best_config = {
        "pipeline_config_id": best_row["pipeline_config_id"],
        "data": string_to_dict(best_row["data"]),
        "train": string_to_dict(best_row["train"]),
    }
    print("Best Config for Retraining: ", best_config)
    
    # Upgrade Thinning for Eval
    best_config["train"]["model_config"]["thinning"] = deepcopy(cheap_thinning)
    for split in ["train_dir", "valid_dir", "test_dir"]:
        best_config["data"][ds_name][split] = f"ddrg/MUSES/{ds_name}" # Use full dataset for retraining and evaluation
    best_config["train"]["trainer_config"]["valid_freq"] = 1 # for early stopping
    best_config["data"][ds_name].pop("test_dir", None) # EasyTPP evals test set after each epoch (Not Good)
  

    for seed in range(5): # TODO to 5 
        print(f"Retraining with best config, seed {seed}...")
        set_seed(seed)

        # Also adjust seed of the trainer
        best_config["train"]["trainer_config"]["seed"] = seed
        best_config["train"]["base_config"]["base_dir"] = f"../checkpoints/{ds_name}/{model_name}/{seed}" 
        with open(train_config_address, 'w') as f:
            yaml.safe_dump(best_config, f)
        
        runner_config = Config.build_from_yaml_file(str(train_config_address), experiment_id="train")

        model_runner = Runner.build_from_config(runner_config)

        model_runner.run()

        #ing Eval is more expensive than training Due to Thinn --> Reduce Other consumption to compensate
        best_config_eval = deepcopy(best_config)
        best_config_eval["train"]["model_config"]["thinning"] = deepcopy(expensive_thinning)
        best_config_eval["train"]["base_config"]["stage"] = "eval"
        best_config_eval["data"][ds_name]["test_dir"] = f"ddrg/MUSES/{ds_name}" 
        best_config_eval["train"]["trainer_config"]["max_epoch"] = 1 
        best_config_eval["train"]["trainer_config"]["batch_size"] = 8
        best_config_eval["train"]["model_config"]["pretrained_model_dir"] =  model_runner.runner_config.base_config.specs['saved_model_dir']
        best_config_eval["eval"] = best_config_eval["train"] # Lets call experiment_id "eval" now
        best_config_eval.pop("train")
        with open(eval_config_address, 'w') as f:
            yaml.safe_dump(best_config_eval, f)

        runner_config_eval = Config.build_from_yaml_file(str(eval_config_address), experiment_id="eval")
        model_runner_eval = Runner.build_from_config(runner_config_eval)


        valid_results = model_runner_eval.evaluate(return_all_metrics=True)
        test_results = model_runner_eval.evaluate(model_runner_eval._data_loader.test_loader(), return_all_metrics=True) 
        train_results = model_runner_eval.evaluate(model_runner_eval._data_loader.train_loader(), return_all_metrics=True) # To check for overfitting

        results = deepcopy(best_config)
        results["seed"] = seed
        results["dataset"] = ds_name
        results["model"] = model_name
        results["search_id"] = str(search_id)
        results["search_duration"] = search_duration
        for split, r in[("valid", valid_results), ("test", test_results), ("train", train_results)]:
            for metric_name, metric_value in r.items():
                results[f"{split}_{metric_name}"] = metric_value
        print("Final Results: ", results)
        store_complex_dict(results, database_path="../results.db", table_name="final_results")
        print(f"Seed {seed} Finished!")
    
    # Mark as finished
    with open(done_file, "a", encoding="utf-8") as f:
        f.write(f"{model_name}|{ds_name}\n")
    print(f"Finished {model_name} on {ds_name}!")



if __name__ == "__main__":
    app()