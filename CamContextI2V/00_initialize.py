import os
import sys
import subprocess
from pathlib import Path
import argparse
from termcolor import colored
import yaml
import datetime

from utils.meta import EXPERIMENT_DIRECTORIES, DEFAULT_META_FILE

parser = argparse.ArgumentParser(description="Script to initialize a new experiment")
parser.add_argument("--config", "-c", default = None, type=str)
parser.add_argument("--name", "-n", default = None, type=str)
parser.add_argument("--machine", "-m", default = "cvg28", type=str)
parser.add_argument("--meta-file", type=str, default=DEFAULT_META_FILE, help="Path to the metadata file")
args = parser.parse_args()

if __name__ == '__main__':
    assert args.config is not None and os.path.exists(args.config), f"Config file '{args.config}' does not exist!"
    assert args.name is not None, "Please provide a name for the experiment!"
    assert args.machine in EXPERIMENT_DIRECTORIES, f"Provided unknown machine '{args.machine}'!"

    exp_dir = Path(EXPERIMENT_DIRECTORIES[args.machine]).resolve()
    run_dir = exp_dir / args.name
    os.makedirs(run_dir, exist_ok=True)

    config_out_file = run_dir / "config.yaml"

    subprocess.run(f"cp {args.config} {config_out_file}", shell=True)

    try:
        with open(args.meta_file, "r") as f:
            meta_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Failed loading meta file '{args.meta_file}': {e}")
        sys.exit(1)
    if meta_data is None: meta_data = {}
    if args.machine not in meta_data: meta_data[args.machine] = {}

    meta_data[args.machine][args.name] = {
        "config_file": str(config_out_file),
        "experiment_directory": str(run_dir),
        "stdout_file": "",
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open(args.meta_file, "w") as f:
            yaml.safe_dump(meta_data, f)
    except Exception as e:
        print(f"Error: Failed loading meta file '{args.meta_file}': {e}")
        sys.exit(1)


    print(colored(f"Successfully initialized new experiment at '{run_dir}'!", "green"))
