import argparse
import os
import textwrap
import yaml
from pathlib import Path
import datetime
import subprocess
import time
import socket
from termcolor import colored
import sys
from utils.meta import *


def get_partitions() -> list:
    """
    Retrieve a list of partition names from the PARTITION_SETUPS configuration.

    Returns:
        list: A list of partition names aggregated from PARTITION_SETUPS.
    """
    partitions = []
    for k, v in PARTITION_SETUPS.items():
        partitions.extend(list(v.keys()))
    return partitions

def arguments():
    """
    Parse and validate command-line arguments for configuring the training job.

    This function sets up the argument parser with various options for configuring
    the experiment, including machine, partition, run name, GPUs, nodes, iterations,
    and other training and scheduling parameters. It also sets default values and
    performs basic validations.

    Returns:
        Namespace: Parsed command-line arguments with additional modifications based on defaults and validations.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config_file", type=str, help="Path to the configuration file")
    parser.add_argument("-m", "--machine", choices=["default", "cvg28"], default=None, help="Machine name (choices: 'marvin', 'bender')")
    parser.add_argument("-p", "--partition", type=str, choices=get_partitions(), default=None)
    parser.add_argument("-r", "--run-name", type=str, help="Name for the job")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations")
    parser.add_argument("--experiment-base", type=str, default=None, help="Base directory for experiments")
    parser.add_argument("--config-out", type=str, default=DEFAULT_CONFIG_OUTPUT, help="Base directory to output the generated config files")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--remote", action="store_true", default=False, help="Run on a remote machine")
    parser.add_argument("--deploy", action="store_true", default=DEFAULT_DEPLOY, help="Deploy the source code if running remotely")
    parser.add_argument("--mem-per-cpu", type=int, default=16, help="Memory per CPU (in GB)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name to continue training from")
    parser.add_argument("--meta-file", type=str, default=DEFAULT_META_FILE, help="Path to the metadata file")
    parser.add_argument("--experiment-directory", type=str, default=None, help="Path to the experiment directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--continue", dest="continue_training", action="store_true", default=False, help="Continue training from the provided or last checkpoint")
    parser.add_argument("--continue-and-fix", dest="continue_training_and_fix", action="store_true", default=False, help="Continue training from the provided or last checkpoint")
    parser.add_argument("--nodelist", type=str, default=None, help="Nodes to run the slurm job on")
    parser.add_argument("--exclude-nodes", type=str, default=None, help="Nodes to exclude from scheduling")
    parser.add_argument("--dependency", type=str, default=None, help="Job dependency")
    parser.add_argument("--run", action="store_true", default=False, help="Directly run and do not use Slurm job scheduling")
    parser.add_argument("--wait-time", type=int, default=None, help="Wait time until the job is started")
    parser.add_argument("--accumulate-grad-batches", type=int, default=None, help="Number of batches to accumulate the gradient over")
    parser.add_argument("--time", type=str, default=None, help="Slurn time limit in hh:mm:ss format")

    args = parser.parse_args()
    
    args.machine = args.machine if args.machine is not None else socket.gethostname()
    args.run_name = args.run_name if args.run_name is not None else str(Path(args.config_file).stem)
    #args.debug = args.partition is None if not args.debug else args.debug
    args.partition = DEFAULT_PARTITION[args.machine] if args.partition is None else args.partition
    args.experiment_base = EXPERIMENT_DIRECTORIES[args.machine] if args.experiment_base is None else args.experiment_base
    args.num_gpus = PARTITION_SETUPS[args.machine][args.partition]["num_gpu"] if args.num_gpus is None else args.num_gpus

    if args.config_file is None or args.checkpoint is not None:
        if args.continue_training or args.continue_training_and_fix:
            assert args.checkpoint is not None, "If not providing a config file, please provide a checkpoint."
            assert args.run_name is not None, "If not providing a config file, please provide a run name."
            try:
                with open(args.meta_file, "r") as f:
                    meta_data = yaml.safe_load(f)
            except Exception as e:
                print(f"Error: Failed loading meta file '{args.meta_file}': {e}")
                sys.exit(1)
            
            if args.run_name not in meta_data[args.machine]:
                print(f"Error: Run name '{args.run_name}' does not exist in the meta data.")
                sys.exit(1)
            
            exp_meta_data = meta_data[args.machine][args.run_name]

            exp_dir = Path(exp_meta_data["experiment_directory"])
            args.config_file = str(exp_dir/"config.yaml")

            ckpt_name = Path(args.checkpoint)
            if ckpt_name.is_absolute():
                args.checkpoint = str(ckpt_name)
            elif len(ckpt_name.parts) > 1:
                args.checkpoint = exp_dir / "checkpoints" / ckpt_name.parts[-1]
            else:
                if ckpt_name.suffix == '':
                    ckpt_name = str(ckpt_name) + ".ckpt"
                args.checkpoint = exp_dir / "checkpoints" / "trainstep_checkpoints" / ckpt_name
            
            if not args.continue_training_and_fix:
                args.experiment_directory = str(exp_dir)
            else:
                args.run_name = args.run_name + "_cont"
            print(f"Continue training from checkpoint '{args.checkpoint}'")
            print(f"Loading config file: {args.config_file}")

    if args.partition.split("_")[-1] == "devel": args.debug = True

    return args

def format_run_script(
        machine: str,
        run_name: str,
        num_gpus: int,
        partition: str,
        run_time: str,
        config_file: str,
        experiment_directory: str,
        script_path: str = "CamContextI2V/main/trainer.py",
        cpus_per_task: int = 2,
        mem_per_cpu: int = 16,
        nodes: int = 1,
        stdout_file: str = "results/stdout.out",
        slurm: bool = True,
        checkpoint: str = None,
        nodelist: str = None,
):
    """
    Format and generate a shell script for running the training job.

    Args:
        machine (str): The machine name on which the script will run.
        run_name (str): The name of the job.
        num_gpus (int): The number of GPUs to be used.
        partition (str): The partition on the machine.
        run_time (str): The time limit for the job in hh:mm:ss format.
        config_file (str): Path to the configuration file.
        experiment_directory (str): Directory for experiment outputs.
        script_path (str, optional): Path to the training script. Defaults to "CamContextI2V/main/trainer.py".
        cpus_per_task (int, optional): Number of CPUs per task. Defaults to 2.
        mem_per_cpu (int, optional): Memory per CPU in GB. Defaults to 16.
        nodes (int, optional): Number of nodes to be used. Defaults to 1.
        stdout_file (str, optional): File path for stdout output. Defaults to "results/stdout.out".
        slurm (bool, optional): Flag indicating whether to format the script for SLURM. Defaults to True.
        checkpoint (str, optional): Path to a checkpoint file if resuming training. Defaults to None.
        nodelist (str, optional): Specific nodes to run the SLURM job on. Defaults to None.

    Returns:
        str: The complete run script as a string.
    """
    run_script = ""
    hashbang = "#!/bin/bash"
    slurm_cmds = textwrap.dedent(f"""\
        #SBATCH --partition={partition}
        #SBATCH --account ag_ifi_gall
        #SBATCH --job-name={run_name}
        #SBATCH --output={stdout_file}
        #SBATCH --error={stdout_file}
        #SBATCH --cpus-per-task={cpus_per_task}            
        #SBATCH --ntasks={num_gpus}
        #SBATCH --ntasks-per-node={num_gpus}
        #SBATCH --mem-per-cpu={mem_per_cpu}G             
        #SBATCH --nodes={nodes}
        #SBATCH --gpus={num_gpus}
        #SBATCH --time={run_time}   
    """)
    env_setup_cmd = ENVIRONMENT_SETUP[machine]
    
    if checkpoint is None:
        run_cmd = textwrap.dedent(f"""\
            torchrun --standalone --nproc_per_node={num_gpus} --node_rank=0 --rdzv_id=12345 --rdzv_backend=c10d {SOURCE_PATH[machine]}/{script_path} \\
                --name {run_name} \\
                --base {config_file} \\
                --logdir {experiment_directory} \\
                --train                    
        """)
    else:
        run_cmd = textwrap.dedent(f"""\
            torchrun --standalone --nproc_per_node={num_gpus} --node_rank=0 --rdzv_id=12345 --rdzv_backend=c10d {SOURCE_PATH[machine]}/{script_path} \\
                --name {run_name} \\
                --base {config_file} \\
                --load_from_checkpoint {checkpoint} \\
                --logdir {experiment_directory} \\
                --train
                """)
    
    if machine == "cvg28":
        run_cmd = run_cmd[:-2] +  "\\\n\t--cwd /home/denninge/CamContextI2V"

    
    run_script += f"{hashbang}\n"
    run_script += f"{slurm_cmds}\n" if slurm else ""
    run_script += f"{env_setup_cmd}\n"
    run_script += f"{run_cmd}\n"

    return run_script


def format_config_file(
        config_file: str,
        run_name: str,
        experiment_directory: str,
        machine: str,
        num_gpus: int,
        nodes: int = 1,
        num_iterations: int = None,
        batch_size: int = None,
        num_workers: int = None,
        enable_wandb: bool = True,
        checkpoint: str = None,
        accumulate_grad_batches: int = 4,
        debug: bool = False
        ):
    """
    Load and modify the YAML configuration file for training.

    This function reads the configuration file, updates paths for pretrained checkpoints,
    data directories, logging configurations, and various training parameters based on
    the provided arguments.

    Args:
        config_file (str): Path to the original YAML configuration file.
        run_name (str): Name of the training run.
        experiment_directory (str): Directory to store experiment outputs.
        machine (str): The machine name to determine data and environment paths.
        num_gpus (int): Number of GPUs to use.
        nodes (int, optional): Number of nodes to use. Defaults to 1.
        num_iterations (int, optional): Number of training iterations. Defaults to None.
        batch_size (int, optional): Batch size for training. Defaults to None.
        num_workers (int, optional): Number of workers for data loading. Defaults to None.
        enable_wandb (bool, optional): Flag to enable or disable wandb logging. Defaults to True.
        checkpoint (str, optional): Path to a checkpoint file if resuming training. Defaults to None.
        accumulate_grad_batches (int, optional): Number of batches to accumulate gradients over. Defaults to 4.
        debug (bool, optional): Flag to enable debug mode settings. Defaults to False.

    Returns:
        dict: Modified configuration dictionary with updated parameters.
    """
    
    with open(config_file, "r") as f:
        config_file = yaml.safe_load(f)
    if checkpoint is None:
        pretrained_path = str(Path(SOURCE_PATH[machine]) / config_file["model"]["pretrained_checkpoint"])
    else:
        pretrained_path = checkpoint
    config_file["model"]["pretrained_checkpoint"] = pretrained_path

    config_file["data"]["params"]["train"]["params"]["data_dir"] = DATA_DIRECTORIES[machine]["train"]["data_dir"]
    config_file["data"]["params"]["train"]["params"]["meta_path"] = DATA_DIRECTORIES[machine]["train"]["meta_path"]
    config_file["data"]["params"]["train"]["params"]["meta_list"] = DATA_DIRECTORIES[machine]["train"]["meta_list"]
    config_file["data"]["params"]["train"]["params"]["caption_file"] = DATA_DIRECTORIES[machine]["train"]["caption_file"]

    config_file["data"]["params"]["validation"]["params"]["data_dir"] = DATA_DIRECTORIES[machine]["val"]["data_dir"]
    config_file["data"]["params"]["validation"]["params"]["meta_path"] = DATA_DIRECTORIES[machine]["val"]["meta_path"]
    config_file["data"]["params"]["validation"]["params"]["meta_list"] = DATA_DIRECTORIES[machine]["val"]["meta_list"]
    config_file["data"]["params"]["validation"]["params"]["caption_file"] = DATA_DIRECTORIES[machine]["val"]["caption_file"]

    if batch_size is not None:
        config_file["data"]["params"]["batch_size"] = batch_size
    if num_workers is not None:
        config_file["data"]["params"]["num_workers"] = num_workers
    if machine in ["marvin","bender"]:
        config_file["lightning"]["trainer"]["enable_progress_bar"] = False

    config_file["lightning"]["trainer"]["devices"] = num_gpus
    config_file["lightning"]["trainer"]["num_nodes"] = nodes
    config_file["lightning"]["trainer"]["accumulate_grad_batches"] = accumulate_grad_batches
    if "progress_printer" in config_file["lightning"]["callbacks"]:
        config_file["lightning"]["callbacks"]["progress_printer"]["params"]["accumulate_grad_batches"] = accumulate_grad_batches

    if num_iterations is not None:
        config_file["lightning"]["params"]["max_steps"] = num_iterations

    config_file["lightning"]["logger"] = {
        "target": "pytorch_lightning.loggers.wandb.WandbLogger",
        "params": 
            {
                "project": "camcontexti2v",
                "name": run_name,
                "save_dir": str(experiment_directory),
                "log_model": False,
                "mode": "online" if enable_wandb else "disabled"
            }
    }
    if debug:
        config_file["lightning"]["callbacks"]["model_watcher"] = {
            "target": "callbacks.ModelWatcherCallback",
            "params": {
                "max_steps": 200,
                "log_dir": 'debug'
            }
        }

    return config_file

def write_exp_meta_file(
        machine: str,
        run_name: str,
        config_file: str,
        experiment_directory: str,
        stdout_file: str,
        filename: str = ".exp_meta.yaml"):
    """
    Write experiment metadata to a YAML file.

    This function updates (or creates) a metadata file with information about the current
    experiment, including the configuration file path, experiment directory, stdout log file,
    and the current timestamp.

    Args:
        machine (str): The machine name where the experiment is running.
        run_name (str): Name of the experiment run.
        config_file (str): Path to the configuration file used for the experiment.
        experiment_directory (str): Directory where experiment outputs are stored.
        stdout_file (str): File path for the standard output log.
        filename (str, optional): Path to the metadata file. Defaults to ".exp_meta.yaml".

    Returns:
        None
    """
    meta = {}
    if os.path.exists(filename):
        meta = yaml.safe_load(open(filename, "r"))

    if machine not in meta:
        meta[machine] = {}

    config_file_abs = str(Path(config_file).absolute())
    experiment_directory_abs = str(Path(experiment_directory).absolute())
    stdout_file_abs = str(Path(stdout_file).absolute())

    meta[machine][run_name] = {
        "config_file": config_file_abs,
        "experiment_directory": experiment_directory_abs,
        "stdout_file": stdout_file_abs,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(filename, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)
    return

def fix_checkpoint(checkpoint):
    """
    Convert a DeepSpeed checkpoint to a collapsed checkpoint if necessary.

    This function checks if the provided checkpoint file has the expected file extension.
    If not, it uses the DeepSpeed utility to convert the checkpoint to a format with a ".pt" extension.

    Args:
        checkpoint (str or Path): Path to the checkpoint file.

    Returns:
        Path or str: The original checkpoint if it is already a ".pt" file, or the path to the converted checkpoint.
    """
    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".pt":
        return checkpoint
    
    print("Converting DeepSpeed checkpoint to collapsed checkpoint")
    ckpt_name = checkpoint.stem
    ckpt_path = checkpoint.parent
    output_path = str(ckpt_path / (str(ckpt_name) + ".pt"))
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint, output_path)
    return output_path

def main():
    """
    Main function to set up and execute the training job.

    This function parses command-line arguments, sets up the experiment directory,
    configures the run and configuration files, writes experiment metadata, and then
    either submits a SLURM job or directly runs the training script based on the configuration.

    Returns:
        None
    """
    args = arguments()

    if args.remote: 
        print(colored(f"Running remotely on {args.machine}", "yellow"))
        if args.deploy:
            print(colored(f"Deploying source code to {args.machine}", "yellow"))
            deploy_cmd = f"source ~/.bash_aliases && deploy {HOSTNAMES[args.machine]}"
            subprocess.run(deploy_cmd, shell=True, executable="/bin/bash")
        line_args = sys.argv
        for i, a in enumerate(line_args):
            if a == "--remote":
                del line_args[i]
        ssh_cmd = f"ssh {HOSTNAMES[args.machine]} 'cd CamContextI2V && conda activate cami2v && python {' '.join(line_args)}'"
        print(f"Running: {ssh_cmd}")
        subprocess.run(ssh_cmd, shell=True)
        return
    
    is_slurm = args.machine in ["marvin", "bender", "lamarr"] and not args.run
    if args.experiment_directory is None:
        experiment_directory = Path(args.experiment_base) / ((f"{args.run_name}_debug" if args.debug else args.run_name) + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
        config_name = str(Path(args.config_file).stem)
        config_out_file_name = f"{config_name}_{args.machine}"
        config_out_file_name = config_out_file_name + "_debug" if args.debug else config_out_file_name
        config_out_file_name += ".yaml"
        config_out_file_path = str(Path(args.config_out)/config_out_file_name)
    else:
        experiment_directory = Path(args.experiment_directory)
        config_out_file_path = args.config_file

    stdout_file = f"results/{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.out"

    if not os.path.exists(args.config_out):
        os.makedirs(args.config_out)
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    run_script_file = experiment_directory/"train.slurm" if is_slurm else experiment_directory/"train.sh"

    run_script = format_run_script(
        machine = args.machine,
        run_name = args.run_name,
        num_gpus = args.num_gpus,
        partition = args.partition,
        run_time = args.time or PARTITION_SETUPS[args.machine][args.partition]["run_time"],
        config_file = config_out_file_path,
        experiment_directory = str(experiment_directory),
        script_path = "CamContextI2V/main/trainer.py",
        nodes = args.num_nodes,
        stdout_file=stdout_file,
        slurm = is_slurm,
        mem_per_cpu=args.mem_per_cpu,
        checkpoint=args.checkpoint if args.continue_training else None,
    )

    if args.continue_training_and_fix or (not args.continue_training and args.checkpoint is not None and Path(args.checkpoint).suffix == ".ckpt"):
        args.checkpoint = fix_checkpoint(args.checkpoint) if not args.continue_training else args.checkpoint
    
    config_file = format_config_file(
        config_file = args.config_file,
        run_name = args.run_name,
        experiment_directory=str(experiment_directory),
        machine = args.machine,
        num_gpus = args.num_gpus,
        nodes = args.num_nodes,
        num_iterations = args.iterations,
        batch_size = PARTITION_SETUPS[args.machine][args.partition]["batch_size"] if args.batch_size is None else args.batch_size,
        num_workers = 1 if args.debug else None,
        enable_wandb = not args.debug,
        checkpoint = args.checkpoint if not args.continue_training else None,
        accumulate_grad_batches=PARTITION_SETUPS[args.machine][args.partition]["accumulate_grad_batches"] if args.accumulate_grad_batches is None else args.accumulate_grad_batches,
        debug = args.debug
    )

    write_exp_meta_file(
        machine = args.machine,
        run_name = args.run_name,
        config_file = config_out_file_path,
        stdout_file = stdout_file,
        experiment_directory = str(experiment_directory))
    print(f"Updated experiment metadata.")

    with open(config_out_file_path, "w") as f:
        yaml.dump(config_file, f, default_flow_style=False, sort_keys=False)
    with open(experiment_directory/"config.yaml", "w") as f:
        yaml.dump(config_file, f, default_flow_style=False, sort_keys=False)
    with open(run_script_file, "w") as f:
        f.write(run_script)

    print(f"Run script: {str(run_script_file)}")
    print(f"Config file: {config_out_file_path}")
    print(f"Experiment directory: {str(experiment_directory)}")

    if is_slurm:
        run_cmd = ["sbatch"]
        if args.dependency is not None: run_cmd += [f"--dependency={args.dependency}"]
        if args.exclude_nodes is not None: run_cmd += [f"--exclude={args.exclude_nodes}"]; print(f"Exclude nodes: {args.exclude_nodes}")
        if args.nodelist is not None: run_cmd += [f"--nodelist={args.nodelist}"]; print(f"Node list: {args.nodelist}")
        if args.wait_time is not None: run_cmd += [f"--begin=now+{args.wait_time}hour"]
        run_cmd += [str(run_script_file)]

        result = subprocess.run(run_cmd, capture_output=True)

        if result.returncode == 0:
            print(colored("Job submitted successfully.", "green"))
            print(result.stdout.decode('utf-8') + "\n")
        else:
            print(colored("Error submitting job:\n", "red"), result.stderr.decode('utf-8'))

        time.sleep(2)
        subprocess.run(["squeue", "--me"])
        print(colored("Training started successfully.", "green"))
    else:
        print(colored("Starting training script...", "green"))
        os.execv("/bin/bash", ["bash", str(run_script_file)])

if __name__ == "__main__":
    main()
    sys.exit(0)
