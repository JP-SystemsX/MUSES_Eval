from copy import deepcopy
import datetime
import json
import numpy as np
from pathlib import Path
import hashlib
import uuid
import yaml
from time import sleep
import sqlite3
from ast import literal_eval
import torch
import random
import pandas as pd
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
from time import time
import gc

def string_to_dict(s):
    if isinstance(s, str) and s.startswith("{") and s.endswith("}"):
        result = {}
        for key, value in json.loads(s).items():
            result[key] = string_to_dict(value)
        return result
    elif isinstance(s, str) and s.startswith("[") and s.endswith("]"):
        return literal_eval(s)
    return s


def make_dict_storable(advanced_dictionary: dict)->dict:
    """
    Takes in a dict with advanced values like datetime, dicts, lists, etc. 
    and converts it into a dict that can be stored in an sqlite database meaning strings, numbers, etc.

    Args:
        advanced_dictionary (dict): dict with potentially complex datatypes

    Returns:
        dict: A dictionary with only simple datatypes
    """
    simple_dict = {}
    for key, value in advanced_dictionary.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, (int, float, str)):
            pass
        elif isinstance(value, (bool, np.bool_)):
            value = 1 if value else 0
        elif isinstance(value, (bytes, Path, list)) or value is None:
            value = str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            value = value.strftime("%d-%m-%Y %H:%M:%S")
        elif isinstance(value, dict):
            value = json.dumps(make_dict_storable(value))
        #elif isinstance(value, str):
        else:
            raise NotImplementedError(f"dtype {type(value)} is currently not storable, but can likely be easily added in make_dict_storable()")
        simple_dict[key] = value

    return simple_dict


def hash_dict(d: dict) -> str:
    """Coverts Simple Dict into hash 

    Args:
        d (dict): Dict that shall be hashed

    Returns:
        str: hashcode
    """
    # Not sure how more complex types get hashed (often represented as some kind of RANDOM id)
    # Hence Simplify first  
    d = make_dict_storable(d)  
    # Make Consistent ~ Convert dict to a sorted tuple 
    dict_tuple = tuple(sorted(d.items()))
    # Convert to binaries
    dict_string = str(dict_tuple).encode()
    # Hash binaries
    hash_object = hashlib.sha256(dict_string)
    
    return hash_object.hexdigest()


def store_complex_dict(d:dict, database_path: str | Path, table_name: str) -> str:
    storable_dictionary = make_dict_storable(d)
    hash = hash_dict(storable_dictionary)

    create_sqlite_table_from_dict(
        database_path=database_path,
        table_name=table_name,
        primary_keys=["hash_key"],#, "data_config_hash"],
        data_dict={
            # Both Hashs build Primary Key
            "hash_key": hash, 
            **storable_dictionary
        }
    ) 
    return hash



def create_sqlite_table_from_dict(database_path: str | Path, table_name: str, data_dict: dict, primary_keys=None):
    """
    Create a SQLite database and table from a dictionary.

    Args:
        database_path (str): Path to the SQLite database.
        table_name (str): Name of the table to create.
        data_dict (dict): Dictionary where each key is a column name and its value determines the column type.
    """
    # Infer column types based on dictionary values
    def infer_sql_type(value):
        if isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, str):
            return "TEXT"
        elif isinstance(value, bytes):
            return "BLOB"
        else:
            return "TEXT"  # Default to TEXT if the type is unknown
    
    if isinstance(database_path, str):
        database_path = Path(database_path)

    # Connect to SQLite database (create if it doesn't exist)
    database_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

    if (database_path.exists() and database_path.is_dir()) or ((not database_path.exists()) and database_path.suffix == ""):
        database_path.mkdir(exist_ok=True, parents=True)
        # Cache Transaction First
        transaction_id = str(uuid.uuid4())
        transaction_adr = database_path / f"{transaction_id}.yml"
        transaction_data = {
            "table_name": table_name,
            "data_dict": data_dict,
            "primary_keys": primary_keys
        }
        with open(transaction_adr, "w+") as file:
            yaml.safe_dump(transaction_data, file)

        return

    # Build the CREATE TABLE statement
    columns = [f"{key} {infer_sql_type(value)}" for key, value in data_dict.items()]# if key not in primary_keys]
    if not primary_keys:
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns)}
        );
        """
    else:
        #primary_columns = [f"{key} {infer_sql_type(data_dict[key])} PRIMARY KEY" for key in primary_keys]
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)},
            PRIMARY KEY ({', '.join(primary_keys)})
        );
        """

    # Insert the dictionary as a row into the table
    columns_str = ', '.join(data_dict.keys())
    placeholders = ', '.join(['?' for _ in data_dict.values()])
    insert_sql = f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders});"


    # Active Waiting to connect
    i = 0
    while True:
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            # Execute the CREATE TABLE statement
            cursor.execute(create_table_sql)
            cursor.execute(insert_sql, tuple(data_dict.values()))

            conn.commit()
            conn.close()
            break
        except Exception as e:
            print(e, i)
            i += 1
            sleep(0.1)
            if i > 100:
                break


def to_builtin(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)



def trial(
        hyperparameters: dict, 
        fidelity: int,
        seed: int,
        model_name: str,
        ds_name: str,
        search_id: str,
        config_: dict,
        ):
    config = deepcopy(config_)

    hp = dict(hyperparameters)
    thinning_conf = {
        "num_seq": 3,
        "num_sample": 1,
        "num_exp": 4, 
        "look_ahead_time": 4,
        "patience_counter": 5,
        "over_sample_rate": 5,
        "num_samples_boundary": 5,
        "dtime_max": 5
    }

    # Define Fidelity Levels
    fidelity = int(fidelity)
    match fidelity:
        case 3: # Low Fidelity 
            max_seq_cnt = 50_000

        case 9: # Medium Fidelity
            max_seq_cnt = 250_000
            thinning_conf["num_seq"] = 5
            thinning_conf["num_exp"] = 50
            thinning_conf["look_ahead_time"] = 6

        case 27: # High Fidelity
            max_seq_cnt = 1_000_000
            thinning_conf["num_seq"] = 10
            thinning_conf["num_exp"] = 500
            thinning_conf["look_ahead_time"] = 10

    # Format Config
    hp = {k: to_builtin(v) for k, v in hp.items()}
    model_specs = {key.rsplit("model_specs.")[-1]: value for key, value in hp.items() if key.startswith("model_specs.")}
    model_cfg = {key.rsplit("model.")[-1]: value for key, value in hp.items() if key.startswith("model.")}
    model_cfg["model_specs"] = model_specs
    trainer_cfg = {key.rsplit("train.")[-1]: value for key, value in hp.items() if key.startswith("train..")}
    trainer_cfg["metrics"] = ['acc', 'rmse', 'f1_macro', "r2"] 
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
    config["data"][ds_name]["train_dir"] = f"ddrg/MUSES/{ds_name}/{max_seq_cnt}" # Limit Train Set Size based on Fidelity Level
    print("Config for this trial: ", config)

    config_address = Path(f'../tmp/{model_name}-{ds_name}-{search_id}.yaml') # TODO change for parallel 
    with open(config_address, 'w') as f:
            yaml.safe_dump(config, f)


    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    runner_config = Config.build_from_yaml_file(str(config_address), experiment_id="train")


    start_time = time()
    model_runner = Runner.build_from_config(runner_config)

    model_runner.run()
    eval_results = model_runner.evaluate(return_all_metrics=True)

    # Log results to DB
    config["search_id"] = str(search_id)
    config.update(eval_results)
    config["trial_time"] = time() - start_time
    config["timestamp"] = datetime.datetime.now().isoformat()
    config["dataset"] = ds_name
    config["model"] = model_name
    store_complex_dict(config, database_path="../results.db", table_name="trials")

    results = {
        "fitness": -1 * eval_results["acc"] - eval_results["f1_macro"] - 2 * max(0, eval_results["r2"]), 
        "cost": config["trial_time"],
        "info": None
    }
    return results
