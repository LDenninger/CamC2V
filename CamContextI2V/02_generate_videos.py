import argparse
import os
import sys
import textwrap
import yaml
from pathlib import Path
import datetime
import subprocess
import time
import copy
import socket
from termcolor import colored
import shutil
import re
from typing import List

from utils.meta import *


def get_partitions() -> list:
    """
    Retrieve a list of partition names from the PARTITION_SETUPS configuration.

    Returns:
        list: A list containing partition names aggregated from PARTITION_SETUPS.
    """
    partitions = []
    for k, v in PARTITION_SETUPS.items():
        partitions.extend(list(v.keys()))
    return partitions

def arguments():
    """
    Parse and validate command-line arguments for evaluation configuration.

    This function sets up the argument parser with various options required for evaluation,
    including run name, machine, partition, checkpoint, GPU configuration, and other parameters.
    It also processes and updates some of the arguments based on default values and experiment metadata.

    Returns:
        Namespace: Parsed command-line arguments with additional defaults and validations applied.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(dest="run_name", type=str, help="Run name to evaluate.")
    parser.add_argument("-m", "--machine", choices=["marvin", "bender", "cvg28", "lamarr", "cvg27"], default=None, help="Machine name (choices: 'marvin', 'bender', 'cvg28', 'lamarr', 'cvg27')")
    parser.add_argument("-p", "--partition", type=str, choices=get_partitions(), default=None)
    #parser.add_argument("-r", "--run-name", type=str, help="Name for the job")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of Images")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of data workers")
    parser.add_argument("--video-length", type=int, default=16, help="Video length")
    parser.add_argument("--experiment-base", type=str, default=None, help="Base directory for experiments")
    parser.add_argument("-c", "--config-file", type=str, default=None, help="Specific config file to use.")
    parser.add_argument("--config-out", type=str, default=DEFAULT_CONFIG_OUTPUT, help="Base directory to output the generated config files")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--meta-file", type=str, default=DEFAULT_META_FILE, help="Path to the metadata file")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--deploy", action="store_true", default=DEFAULT_DEPLOY, help="Deploy the source code if running remotely")
    parser.add_argument("--remote", action="store_true", default=False, help="Run on a remote machine")
    parser.add_argument("--continue", dest="continue_generation", action="store_true", default=False, help="Continue a previous generation")
    parser.add_argument("--disable-camera", action="store_true", default=False, help="Explicitely disable camera conditioning")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--time", type=str, default=None, help="Slurn time limit in hh:mm:ss format")
    parser.add_argument("--sstrat", type=str, default="random", help="Sampling technique for the additional context")
    parser.add_argument("--nodelist", type=str, default=None, help="Nodes to run the slurm job on")
    parser.add_argument("--exclude-nodes", type=str, default=None, help="Nodes to exclude from scheduling")
    parser.add_argument("--dependency", type=str, default=None, help="Job dependency")
    parser.add_argument("--run", action="store_true", default=False, help="Directly run and do not use Slurm job scheduling")
    parser.add_argument("--lora-scale", type=float, default=None, help="LoRA scale for injected lora layers")
    parser.add_argument("--name", type=str, default=None, help="Name of the generation folder")
    parser.add_argument("--sample-file", type=str, default=None, help="Sample file")
    parser.add_argument("--frame-stride", type=str, default=8, help="Stride to sample")
    parser.add_argument("--num-cond-frames", type=int, default=None)
    args = parser.parse_args()
    
    args.machine = args.machine if args.machine is not None else socket.gethostname()
    args.run_name = args.run_name if args.run_name is not None else str(Path(args.exp_dir).stem)
    args.debug = args.partition is None if not args.debug else args.debug
    args.partition = DEFAULT_PARTITION[args.machine] if args.partition is None else args.partition
    args.experiment_base = EXPERIMENT_DIRECTORIES[args.machine] if args.experiment_base is None else args.experiment_base
    args.num_gpus = PARTITION_SETUPS[args.machine][args.partition]["num_gpu"] if args.num_gpus is None else args.num_gpus
    args.batch_size = args.batch_size if args.batch_size is not None else PARTITION_SETUPS[args.machine][args.partition]["batch_size"]

    if args.checkpoint:

        if not Path(args.checkpoint).is_absolute():
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
            ckpt_name = Path(args.checkpoint)

            if len(ckpt_name.parts) > 1:
                args.checkpoint = str(exp_dir / "checkpoints" / ckpt_name.parts[-1])
            else:
                if ckpt_name.suffix == '':
                    ckpt_name = str(ckpt_name) + ".ckpt"
                args.checkpoint = str(exp_dir / "checkpoints" / "trainstep_checkpoints" / ckpt_name / "checkpoint" / "mp_rank_00_model_states.pt")

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
):
    """
    Generate a shell script for running the evaluation job.

    Args:
        machine (str): Machine name where the job will run.
        run_name (str): Name of the evaluation run.
        num_gpus (int): Number of GPUs to use.
        partition (str): Partition to run the job on.
        run_time (str): SLURM time limit in hh:mm:ss format.
        config_file (str): Path to the configuration file.
        experiment_directory (str): Directory for experiment outputs.
        script_path (str, optional): Path to the evaluation/training script. Defaults to "CamContextI2V/main/trainer.py".
        cpus_per_task (int, optional): Number of CPUs per task. Defaults to 2.
        mem_per_cpu (int, optional): Memory per CPU in GB. Defaults to 16.
        nodes (int, optional): Number of nodes to use. Defaults to 1.
        stdout_file (str, optional): File path for SLURM output. Defaults to "results/stdout.out".
        slurm (bool, optional): Flag to generate SLURM script. Defaults to True.
        checkpoint (str, optional): Path to checkpoint if resuming evaluation. Defaults to None.

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
                --test                    
        """)
    else:
        run_cmd = textwrap.dedent(f"""\
            torchrun --standalone --nproc_per_node={num_gpus} --node_rank=0 --rdzv_id=12345 --rdzv_backend=c10d {SOURCE_PATH[machine]}/{script_path} \\
                --name {run_name} \\
                --base {config_file} \\
                --load_from_checkpoint {checkpoint} \\
                --logdir {experiment_directory} \\
                --test                    
        """)

    run_script += f"{hashbang}\n"
    run_script += f"{slurm_cmds}\n" if slurm else ""
    run_script += f"{env_setup_cmd}\n"
    run_script += f"{run_cmd}\n"

    return run_script


def format_config_file(
        config_file: str,
        experiment_directory: str,        
        run_name: str,
        machine: str,
        num_gpus: int,
        nodes: int = 1,
        num_samples: int = None,
        batch_size: int = None,
        num_workers: int = None,
        test_save_path: str = None,
        exclude_samples: List[str] = None,
        enable_camera_condition: bool = True,
        video_length: int = 16,
        cfg_scale: float = 7.5,
        checkpoint: str = None,
        context_sampling_strategy: str = "random",
        lora_scale: float = None,
        sample_file: str = None,
        frame_stride: int = 8,
        num_cond_frames: int = None,
):
    """
    Load and modify the YAML configuration file for evaluation.

    This function reads the given configuration file and updates various parameters
    such as paths, data directories, model checkpoint, and evaluation-specific settings.
    It also configures image logging, video parameters, and wandb logger options.

    Args:
        config_file (str): Path to the original YAML configuration file.
        experiment_directory (str): Directory where experiment outputs are stored.
        run_name (str): Name of the evaluation run.
        machine (str): Machine name used for determining paths and settings.
        num_gpus (int): Number of GPUs to be used.
        nodes (int, optional): Number of nodes to use. Defaults to 1.
        num_samples (int, optional): Number of samples to evaluate. Defaults to None.
        batch_size (int, optional): Batch size for evaluation. Defaults to None.
        num_workers (int, optional): Number of workers for data loading. Defaults to None.
        test_save_path (str, optional): Directory path to save test results. Defaults to None.
        exclude_samples (List[str], optional): List of sample names to exclude. Defaults to None.
        enable_camera_condition (bool, optional): Flag to enable camera conditioning. Defaults to True.
        video_length (int, optional): Video length parameter. Defaults to 16.
        cfg_scale (float, optional): Classifier-free guidance scale. Defaults to 7.5.
        checkpoint (str, optional): Path to checkpoint file. Defaults to None.
        context_sampling_strategy (str, optional): Strategy for context sampling. Defaults to "random".
        lora_scale (float, optional): LoRA scale for injected lora layers. Defaults to None.
        sample_file (str, optional): Path to a sample file. Defaults to None.
        frame_stride (int, optional): Frame stride for sampling. Defaults to 8.
        num_cond_frames (int, optional): Number of additional conditioning frames. Defaults to None.

    Returns:
        dict: Modified configuration dictionary with updated evaluation settings.
    """
    with open(config_file, "r") as f:
        config_file = yaml.safe_load(f)

    if checkpoint is None:
        pretrained_path = str(Path(SOURCE_PATH[machine]) / config_file["model"]["pretrained_checkpoint"])
    else:
        pretrained_path = checkpoint
    config_file["model"]["pretrained_checkpoint"] = pretrained_path

    if lora_scale is not None:
        config_file["model"]["params"]["lora_config"] = {
            "lora_scale": lora_scale
        }

    config_file["data"]["params"]["train"]["params"]["data_dir"] = DATA_DIRECTORIES[machine]["train"]["data_dir"]
    config_file["data"]["params"]["train"]["params"]["meta_path"] = DATA_DIRECTORIES[machine]["train"]["meta_path"]
    config_file["data"]["params"]["train"]["params"]["meta_list"] = DATA_DIRECTORIES[machine]["train"]["meta_list"]
    config_file["data"]["params"]["train"]["params"]["caption_file"] = DATA_DIRECTORIES[machine]["train"]["caption_file"]
    config_file["data"]["params"]["validation"]["params"]["video_length"] = video_length

    config_file["data"]["params"]["validation"]["params"]["data_dir"] = DATA_DIRECTORIES[machine]["val"]["data_dir"]
    config_file["data"]["params"]["validation"]["params"]["meta_path"] = DATA_DIRECTORIES[machine]["val"]["meta_path"]
    config_file["data"]["params"]["validation"]["params"]["meta_list"] = DATA_DIRECTORIES[machine]["val"]["meta_list"]
    config_file["data"]["params"]["validation"]["params"]["caption_file"] = DATA_DIRECTORIES[machine]["val"]["caption_file"]
    config_file["data"]["params"]["validation"]["params"]["caption_file"] = DATA_DIRECTORIES[machine]["val"]["caption_file"]
    config_file["data"]["params"]["validation"]["params"]["video_length"] = video_length
    config_file["data"]["params"]["validation"]["params"]["frame_stride"] = frame_stride
    
    if num_cond_frames is not None:
        config_file["data"]["params"]["validation"]["params"]["num_additional_cond_frames"] = num_cond_frames

    if context_sampling_strategy != 'none':
        config_file["data"]["params"]["validation"]["params"]["additional_cond_frames"] = context_sampling_strategy

    if sample_file is not None:
        config_file["data"]["params"]["validation"]["params"]["meta_list"] = sample_file

    if exclude_samples is not None:
        config_file["data"]["params"]["validation"]["params"]["exclude_samples"] = exclude_samples

    config_file["data"]["params"]["test"] = copy.deepcopy(config_file["data"]["params"]["validation"])

    if num_samples is not None:
        config_file["data"]["params"]["validation_max_n_samples"] = num_samples
        config_file["data"]["params"]["test_max_n_samples"] = num_samples

    if batch_size is not None:
        config_file["data"]["params"]["batch_size"] = batch_size
    if num_workers is not None:
        config_file["data"]["params"]["num_workers"] = num_workers

    config_file["lightning"]["trainer"]["devices"] = num_gpus
    config_file["lightning"]["trainer"]["num_nodes"] = nodes

    config_file["lightning"]["callbacks"]["batch_logger"] = {
        "target": "callbacks.ImageLogger",
        "params":
            {
                "train_batch_frequency": 2500,  # optimization_steps
                "log_first_iteration": True,
                "to_local": True,
                "to_tensorboard": False,
                "to_wandb": False,
                "num_batches": 4,
                "num_val_batches": -1,
                "save_suffix": '',
                "log_all_gpus": True,
                "log_images_kwargs":
                    {
                        "ddim_steps": 25,
                        "ddim_eta": 1.0,
                        "unconditional_guidance_scale": cfg_scale,
                        "timestep_spacing": "uniform_trailing",
                        "guidance_rescale": 0.7,
                        "sampled_img_num": batch_size,
                        "enable_camera_condition": enable_camera_condition,
                    }
            }
    }
    if test_save_path is not None:
        config_file["lightning"]["callbacks"]["batch_logger"]["params"]["test_directory"] = test_save_path

    config_file["lightning"]["logger"] = {
        "target": "pytorch_lightning.loggers.wandb.WandbLogger",
        "params": 
            {
                "project": "camcontexti2v",
                "name": run_name,
                "save_dir": experiment_directory,
                "log_model": False,
                "mode": "disabled"
            }
    }

    config_file["lightning"]["progress_printer"] = {
        "target": "callbacks.PrintProgressCallback",
        "params":
            {            
                "interval": 1,
                "max_steps": 20000,
                "accumulate_grad_batches": 4,
                "print_test_iterations": True
            }
    }
    return config_file

def save_sub_config(self, config, config_file):
    """
    Save a configuration dictionary to a YAML file.

    Args:
        self: Reference to the instance (unused here).
        config (dict): Configuration dictionary to save.
        config_file (str): Path to the file where configuration will be saved.

    Returns:
        None
    """
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return

def write_exp_meta_file(
        machine: str,
        run_name: str,
        config_file: str,
        eval_output: str,
        stdout_file: str,
        filename: str = ".exp_meta.yaml"):
    """
    Update experiment metadata file with evaluation details.

    This function updates the metadata YAML file with information regarding the evaluation,
    such as evaluation output directory, configuration file used for evaluation, stdout file, and timestamp.

    Args:
        machine (str): Name of the machine.
        run_name (str): Evaluation run name.
        config_file (str): Path to the evaluation configuration file.
        eval_output (str): Evaluation output directory.
        stdout_file (str): Path to the standard output file.
        filename (str, optional): Metadata file path. Defaults to ".exp_meta.yaml".

    Returns:
        None
    """
    meta = {}
    if os.path.exists(filename):
        meta = yaml.safe_load(open(filename, "r"))

    if machine not in meta:
        meta[machine] = {}

    eval_output_abs = str(Path(eval_output).absolute())
    eval_config_file_abs = str(Path(config_file).absolute())
    stdout_file_abs = str(Path(stdout_file).absolute())

    meta[machine][run_name].update(
        {
            "eval_output": eval_output_abs,
            "eval_config_file": eval_config_file_abs,
            "eval_stdout_file": stdout_file_abs,
            "eval_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    with open(filename, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)
    return

def write_test_meta_info(
        machine: str,
        run_name: str,
        eval_output: str,
        num_samples: int,
        checkpoint: str,
        test_save_dir: str,
        test_time: str
):
    """
    Write test metadata information to a meta.yaml file within the test save directory.

    Args:
        machine (str): Machine name.
        run_name (str): Evaluation run name.
        eval_output (str): Path to the evaluation output file.
        num_samples (int): Number of samples used for evaluation.
        checkpoint (str): Path to the checkpoint file.
        test_save_dir (str): Directory where test results are saved.
        test_time (str): Timestamp of when the test was executed.

    Returns:
        None
    """
    with open(os.path.join(test_save_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "eval_output": eval_output,
                "num_samples": num_samples,
                "checkpoint": checkpoint,
                "test_save_dir": test_save_dir,
                "machine": machine,
                "run_name": run_name,
                "test_time": test_time
            },
            f,
            default_flow_style=False,
            sort_keys=False
        )

def main():
    """
    Main function to set up and execute the evaluation job.

    This function parses command-line arguments, loads experiment metadata, sets up the evaluation
    configuration and run script, writes updated configuration and metadata, and finally submits
    the evaluation job via SLURM or executes it directly.

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
        ssh_cmd = f"ssh {HOSTNAMES[args.machine]} 'cd CamI2V && conda activate cami2v && python {' '.join(line_args)}'"
        print(f"Running: {ssh_cmd}")
        subprocess.run(ssh_cmd, shell=True)
        return

    is_slurm = args.machine in ["marvin", "bender", "lamarr"] and not args.run
    run_name = args.run_name
    with open(args.meta_file, "r") as f:
        exp_meta_data = yaml.safe_load(f)

    if not run_name in exp_meta_data[args.machine]:
        print(f"Error: Run name {run_name} not found in experiment meta data.")
        sys.exit(1)

    exp_meta_data = exp_meta_data[args.machine][run_name]

    experiment_directory = Path(exp_meta_data["experiment_directory"])
    config_file = exp_meta_data["config_file"]
    
    time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.checkpoint is not None:
        ckpt_parts = Path(args.checkpoint).parts 
        checkpoint_name = [p for p in ckpt_parts if Path(p).suffix == ".ckpt"][0]
    else:
        checkpoint_name = "base"

    config_file = experiment_directory / "config.yaml"
    eval_config_file = experiment_directory / "eval_config.yaml"
    run_script_file = experiment_directory/"eval.slurm" if is_slurm else experiment_directory/"eval.sh"
    stdout_file = f"results/{args.run_name}_eval_{time_now}.out"

    if args.name:
        test_save_dir = experiment_directory / "images" / "test" / args.name
    else:
        test_save_dir = experiment_directory / "images" / "test" / f"{checkpoint_name}_{time_now}"

    exclude_samples = None
    
    if args.continue_generation:
        regex = fr"{checkpoint_name}*"
        regex = re.compile(regex)
        matching_folders = [
            folder for folder in (test_save_dir.parent).iterdir()
            if folder.is_dir() and regex.match(folder.name)
        ]
        
        matching_folder = sorted(matching_folders, key=lambda x: x.name)[-1] if matching_folders else None
        if matching_folder is not None:
            test_save_dir = matching_folder.resolve()
            exclude_samples = [str(p.stem) for p in matching_folder.iterdir() if p.is_dir()]

    os.makedirs(test_save_dir, exist_ok=True)

    run_script = format_run_script(
        machine = args.machine,
        run_name = args.run_name,
        num_gpus = args.num_gpus,
        partition = args.partition,
        run_time = args.time or PARTITION_SETUPS[args.machine][args.partition]["run_time"],
        config_file = eval_config_file,
        experiment_directory = str(experiment_directory),
        script_path = "CamContextI2V/main/trainer.py",
        nodes = args.num_nodes,
        stdout_file=stdout_file,
        slurm = is_slurm,
        #checkpoint = args.checkpoint
    )
    
    config_file = format_config_file(
        config_file = str(config_file),
        run_name = args.run_name,
        experiment_directory = str(experiment_directory),
        machine = args.machine,
        num_gpus = args.num_gpus,
        num_samples = args.num_samples,
        nodes = args.num_nodes,
        batch_size = args.batch_size,
        num_workers = 0 if args.debug else None,
        exclude_samples = exclude_samples,
        test_save_path=str(test_save_dir),
        enable_camera_condition=not args.disable_camera,
        video_length=args.video_length,
        checkpoint = args.checkpoint,
        cfg_scale = args.cfg_scale,
        context_sampling_strategy = args.sstrat,
        sample_file=args.sample_file,
        frame_stride = args.frame_stride,
        num_cond_frames = args.num_cond_frames
    )

    write_exp_meta_file(
        machine = args.machine,
        run_name = args.run_name,
        config_file = str(eval_config_file),
        stdout_file = stdout_file,
        eval_output = str(experiment_directory))
    print(f"Updated experiment metadata.")

    write_test_meta_info(
        machine = args.machine,
        run_name = args.run_name,
        eval_output = stdout_file,
        num_samples = args.num_samples,
        checkpoint = args.checkpoint,
        test_save_dir = str(test_save_dir),
        test_time = time_now
    )

    with open(eval_config_file, "w") as f:
        yaml.dump(config_file, f, default_flow_style=False, sort_keys=False)
    with open(run_script_file, "w") as f:
        f.write(run_script)

    print(f"Run script: {str(run_script_file)}")
    print(f"Config file: {eval_config_file}")
    print(f"Output directory: {str(experiment_directory)}")
    
    if is_slurm:
        run_cmd = ["sbatch"]
        if args.dependency is not None: run_cmd += [f"--dependency={args.dependency}"]
        if args.exclude_nodes is not None: run_cmd += [f"--exclude={args.exclude_nodes}"]; print(f"Exclude nodes: {args.exclude_nodes}")
        if args.nodelist is not None: run_cmd += [f"--nodelist={args.nodelist}"]; print(f"Node list: {args.nodelist}")
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
        print(colored("Starting generation script...", "green"))
        os.execv("/bin/bash", ["bash", str(run_script_file)])

if __name__ == "__main__":
    main()
