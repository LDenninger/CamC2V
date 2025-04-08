import os, re, sys
from omegaconf import OmegaConf
import logging
import atexit
import functools
from termcolor import colored
mainlogger = logging.getLogger('mainlogger')

import torch
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path

from pytorch_lightning.utilities.model_summary import ModelSummary

def init_workspace(logdir, model_config, lightning_config, rank=0):
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    loginfo = os.path.join(logdir, "logs")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))

    mainlogger.info(f"Initialized experiment directory at {logdir}")
    return logdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        #"model_checkpoint": {
        #    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        #    "params": {
        #        "dirpath": ckptdir,
        #        "filename": "{epoch}",
        #        "verbose": True,
        #        "save_last": False,
        #    }
        #},
        #"batch_logger": {
        #    "target": "callbacks.ImageLogger",
        #    "params": {
        #        "save_dir": logdir,
        #        "batch_frequency": 1000,
        #        "max_images": 4,
        #        "clamp": True,
        #    }
        #},    
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
    
    if "batch_logger" in lightning_config.callbacks:
        lightning_config.callbacks["batch_logger"]["params"]["save_dir"] = str(Path(logdir).absolute())
    if 'model_watcher' in lightning_config.callbacks:
        lightning_config.callbacks["model_watcher"]["params"]["log_dir"] = os.path.join(str(Path(logdir).absolute()), lightning_config.callbacks["model_watcher"]["params"]["log_dir"])
    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_debug, wandb_default=True, name='camcontexti2v'):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        "wandb": {
            "target": "pytorch_lightning.loggers.wandb.WandbLogger",
            "params": {
                "project": "cam-i2v-dev",
                "save_dir": logdir,
                "name": name,
                "log_model": True,
            }
        }
    }
    logdir = str(Path(logdir).absolute())
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
    if wandb_default:
        default_logger_cfg = default_logger_cfgs["wandb"]
    else:
        default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")
        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():  # ddp
                try:
                    model.load_state_dict(pl_sd["state_dict"], strict=True)
                except:
                    ## rename the keys for 256x256 model
                    new_pl_sd = OrderedDict()
                    for k, v in pl_sd["state_dict"].items():
                        new_pl_sd[k] = v

                    for k in list(new_pl_sd.keys()):
                        if "framestride_embed" in k:
                            new_key = k.replace("framestride_embed", "fps_embedding")
                            new_pl_sd[new_key] = new_pl_sd[k]
                            del new_pl_sd[k]
                    try:
                        model.load_state_dict(new_pl_sd, strict=True)
                        print('ddp load mode strict=True succeeded')
                    except:
                        model.load_state_dict(new_pl_sd, strict=False)
                        print('ddp load mode strict=False succeeded')
                    del new_pl_sd

                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:  # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    # new_pl_sd[key[16:]]=pl_sd['module'][key]
                    new_pl_sd[key]=pl_sd['module'][key]
                try:
                    model.load_state_dict(new_pl_sd, strict=True)
                    print('deepspeed load mode strict=True succeeded')
                except:
                    model.load_state_dict(new_pl_sd, strict=False)
                    print('deepspeed load mode strict=False succeeded')
        except:
            model.load_state_dict(pl_sd, strict=False)
            print('ddp & deepload load failed, plain load succeeded')

        del pl_sd
    else:
        mainlogger.info(">>> Start training from scratch")

    return model

#def set_logger(logfile, name='mainlogger'):
#    logger = logging.getLogger(name)
#    logger.setLevel(logging.INFO)
#    fh = logging.FileHandler(logfile, mode='w')
#    fh.setLevel(logging.INFO)
#    ch = logging.StreamHandler()
#    ch.setLevel(logging.DEBUG)
#    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
#    ch.setFormatter(logging.Formatter("%(message)s"))
#    logger.addHandler(fh)
#    logger.addHandler(ch)
#    return logger

###=== Output Logging ===###
class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "yellow")
        else:
            return log
        return prefix + " " + log

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io

@functools.lru_cache()
def setup_logger(output_dir:str, name: str='mainlogger', dist_rank:int=0,
                 color:bool=True, log_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s:%(lineno)d %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    abbrev_name = 'AD'
    
    # stdout logging: master only
    if dist_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(log_level)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s:%(lineno)d]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output_dir is not None:
        if output_dir.endswith(".txt") or output_dir.endswith(".log"):
            filename = output_dir
        else:
            filename = os.path.join(output_dir, "log.txt")
        if dist_rank > 0:
            filename = filename + ".rank{}".format(dist_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.FileHandler(filename, 'w')
        
        fh.setLevel(log_level)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger

def cleanup_logging(custom_logger_name='mainlogger'):
    """
    Clean up logging configuration to use only the custom logger.
    
    Args:
        custom_logger_name (str): The name of your custom logger.
    """
    # Get the root logger
    root = logging.getLogger()

    # Remove all handlers from the root logger
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Get your custom logger
    custom_logger = logging.getLogger(custom_logger_name)

    # Set the custom logger as the only child of the root logger
    root.handlers = []
    root.addHandler(logging.NullHandler())  # Add a NullHandler to the root logger
    root.propagate = False

    # Ensure all other loggers propagate to the custom logger
    for name in logging.root.manager.loggerDict:
        if name != custom_logger_name:
            logger = logging.getLogger(name)
            logger.handlers = []
            logger.propagate = True

    # Set the custom logger to not propagate
    custom_logger.propagate = False

    return custom_logger

def human_readable_number(num):
    suffixes = ['','K','M','B','T','P','E']
    
    if abs(num) < 1000:
        return str(num)
    magnitude = 0
    while abs(num) >= 1000:
        num /= 1000.0
        magnitude += 1
    return f"{num:.2f}{suffixes[magnitude]}"

def save_model_summary(model, filename: str, num_spaces:int = 100):
    with open(filename, "w") as f:
        with redirect_stdout(f):
            print(ModelSummary(model, max_depth=-1))  # Print summary to file

        f.write("\n\nTrainable Parameters:\n")
        column_name1 = "Parameter Name"; column_name2 = "#Params"
        f.write((num_spaces+13)*"-")
        f.write(f"\n| {column_name1:<{num_spaces}}| {column_name2}\n")
        f.write((num_spaces+13)*"-")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"\n| {name:<{num_spaces}}| {human_readable_number(param.element_size()* param.numel()):<8}|")
        f.write("\n"+(num_spaces+13)*"-")

def move_tensors_to_cpu(data):
    """
    Recursively move all tensors in a nested structure to CPU and detach them.

    Args:
        data (any): Nested data structure containing tensors (dict, list, tuple, etc.)

    Returns:
        The same data structure with tensors moved to CPU and detached.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, dict):
        return {key: move_tensors_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    elif isinstance(data, set):
        return {move_tensors_to_cpu(item) for item in data}
    # Handle custom objects by trying to access attributes recursively
    elif hasattr(data, '__dict__'):
        for attr in vars(data):
            setattr(data, attr, move_tensors_to_cpu(getattr(data, attr)))
        return data

    return data  # Return non-tensor data unchanged

