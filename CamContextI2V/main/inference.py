import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.path.join(sys.path[0], '../DynDepth-Anything-V2/metric_depth'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import init_workspace, load_checkpoints, setup_logger, cleanup_logging, save_model_summary
import pdb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from lightning.pytorch.profilers import AdvancedProfiler

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")
    parser.add_argument("--config", "-c", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    parser.add_argument("--cwd", type=str, default=None, help="Working directory")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")
    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")

    return parser
    
def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))


if __name__ == "__main__":
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    # parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    
    if args.cwd is not None:
        os.chdir(args.cwd)
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed + global_rank, workers=True)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.config]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())


    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.logdir, config, lightning_config, global_rank)
    logger = setup_logger(loginfo, dist_rank=global_rank)
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    #config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)
    cleanup_logging()
    # print(model)


    
    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    ## update trainer config
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug, name=args.name)

    ## setup callbacks
    callbacks = []
    if "callbacks" in lightning_config:
        callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
        callbacks=[instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    profiler = AdvancedProfiler(filename="perf_logs")
    trainer = Trainer(
        **trainer_config,
        callbacks=callbacks,
        logger=instantiate_from_config(logger_cfg),
        profiler=profiler if args.debug else None
    )
    logger.info(f"Running on {trainer.num_nodes}x{trainer.num_devices} GPUs")


    with torch.autocast('cuda'):
        trainer.test(model, data)