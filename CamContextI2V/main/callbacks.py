import os
import cv2 as cv
import time
import logging
from typing import Any
import wandb
from PIL import Image
import numpy as np
from typing import List
import json
from pathlib import Path
import shutil
from typing import Literal
from omegaconf import OmegaConf
from termcolor import colored
import ipdb


from pytorch_lightning.utilities.types import STEP_OUTPUT

mainlogger = logging.getLogger('mainlogger')

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.save_video import log_local, prepare_to_log, log_evaluation
from utils_train import move_tensors_to_cpu
import pdb

def format_seconds(seconds):
    days = int(seconds // (24 * 3600))
    seconds %= 24 * 3600
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    seconds = int(seconds)
    return f"{days:02}d:{hours:02}h:{minutes:02}m:{seconds:02}s"


class ImageLogger(Callback):
    def __init__(self,
                    train_batch_frequency,
                    images_per_batch: int = -1,
                    num_batches=4, 
                    num_val_batches=4,
                    clamp=True,
                    rescale=True,
                    log_first_iteration=False,
                    save_dir=None, 
                    to_local=False,
                    to_tensorboard=False,
                    to_wandb=True, 
                    log_images_kwargs=None, 
                    log_all_gpus=False,
                    image_directory: str = 'images',
                    test_directory: str = None,
                    keys_to_log = ['image_condition','gt_video','samples'],
                    save_suffix=''):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = train_batch_frequency
        self.num_batches = num_batches
        self.num_val_batches = num_val_batches
        self.images_per_batch = images_per_batch
        self.log_first_iteration = log_first_iteration
        self.log_all_gpus = log_all_gpus
        self.to_local = to_local
        self.to_tensorboard = to_tensorboard
        self.to_wandb = to_wandb
        self.clamp = clamp
        self.keys_to_log = keys_to_log
        self.save_dir = Path(save_dir) / image_directory
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self._cur_log_cnt = 0
        self._cur_mode: Literal["train", "val", "test"] = "none"
        self._log_first_mode = log_first_iteration

        self.test_save_dir = test_directory if test_directory is not None else self.save_dir / "test"
        
        if self.to_local:
            ## default save dir
            if os.path.exists(self.save_dir):
                mainlogger.warning(f"Save directory {self.save_dir} already exists. Overwriting.")
                #shutil.rmtree(self.save_dir)

            os.makedirs(self.save_dir, exist_ok=True)
            config_file = self.save_dir / "sample_config.json"
            save_dict = OmegaConf.to_container(self.log_images_kwargs, resolve=True)
            with open(config_file, 'w') as f:
                json.dump(save_dict, f, indent=4)

            os.makedirs(self.save_dir / "train", exist_ok=True)
            os.makedirs(self.save_dir / "val", exist_ok=True)
            os.makedirs(self.test_save_dir, exist_ok=True)


    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=8):
        """ log images and videos to tensorboard """
        global_step = pl_module.global_step
        for key in batch_logs:
            value = batch_logs[key]
            tag = filename
            if key not in self.keys_to_log:
                continue
            if isinstance(value, list) and isinstance(value[0], str):
                captions = ' |------| '.join(value)
                pl_module.logger.experiment.add_text(tag, captions, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]  # [3, n*h, 1*w]
                grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(tag, grid, fps=save_fps, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n), padding=0)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.experiment.add_image(tag, grid, global_step=global_step)
            else:
                pass

    def log_to_wandb(self, pl_module, batch_logs, filename, split, save_fps=8):
        """ log images and videos to tensorboard """
        global_step = pl_module.global_step
        for key in batch_logs:
            value = batch_logs[key]
            tag = filename + f"/{self._cur_log_cnt}_{key}"
            #if isinstance(value, list) and isinstance(value[0], str):
            #    captions = ' |------| '.join(value)
            #    pl_module.logger.log_text(tag, captions, global_step=global_step)
            if key not in self.keys_to_log:
                continue
            
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                # Video data
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]  # [3, n*h, 1*w]
                grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                video_batch = np.clip((grid*255).numpy(), 0, 255).astype(np.uint8)
                video_batch = [vid for vid in video_batch]
                pl_module.logger.log_video(tag, video_batch, step=global_step)

            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                # image data
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n), padding=0)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.log_image(tag, grid, step=global_step)
            else:
                pass

    def log_batch_imgs(self, trainer, pl_module, batch, batch_idx, split="train"):
        """ generate images, then save and log to tensorboard """

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        filename = "gs{}_ep{}_idx{}_rank{}".format(
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            pl_module.global_rank
        )
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            log_func = pl_module.log_images
            batch_logs = log_func(batch, split=split, **self.log_images_kwargs)

        ## process: move to CPU and clamp
        batch_logs = prepare_to_log(batch_logs, self.images_per_batch, self.clamp)
        torch.cuda.empty_cache()
        
        if self.to_local:
            if split == "test":
                save_dir = self.save_dir / split if split != "test" else self.test_save_dir
                log_evaluation(batch_logs, save_dir, save_fps=7, rescale=True, print_out=True)
            else:
                save_dir = self.save_dir / split / f"step_{str(pl_module.global_step).zfill(6)}" / f"batch_{str(self._cur_log_cnt).zfill(4)}"
                if os.path.exists(save_dir):
                    mainlogger.warning(f"Save directory {save_dir} already exists. Overwriting.")
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                log_local(batch_logs, save_dir, save_fps=7)

        if self.to_tensorboard:
            filename = self.prefix + '_' + filename
            #mainlogger.info("Log [%s] batch <%s> to tensorboard ..." % (split, filename))
            self.log_to_tensorboard(pl_module, batch_logs, filename, split, save_fps=10)

        if self.to_wandb:
            #filename = self.prefix + '_' + filename
            filename = f"{split}/step_{str(pl_module.global_step).zfill(6)}"
            #mainlogger.info("Log [%s] batch <%s> to WandB ..." % (split, filename))
            self.log_to_wandb(pl_module, batch_logs, filename, split, save_fps=10)

        #mainlogger.info(f'Logging {split} batch {self._cur_log_cnt}')
        if is_train:
            pl_module.train()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        if trainer.train_dataloader.dataset.additional_cond_frames in ["random", "random_true"]:
            trainer.train_dataloader.dataset.additional_cond_frames = "random_v2"
        if self.log_first_iteration and pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('train'):
                return
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('train'):
                return
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('val'):
                return
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('test'):
                return
            mainlogger.info(f'[rank {pl_module.global_rank}] Logging batch {batch_idx}')
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="test")

    def _check_batch(self, mode: Literal[Literal['train', 'val', 'test']]):
        """
            Check whether the current training mode: [train, val, test] has changed and reset internal counter.
            Check whether we have reached the maximum number of batches to log for current mode.

            Returns:
                True, if we should log the current batch.
        """
        if mode != self._cur_mode:
            self._cur_log_cnt = 0
        self._cur_log_cnt += 1
        num_batches = self.num_batches if mode == 'train' else self.num_val_batches
        if num_batches >= 0 and self._cur_log_cnt > num_batches:
            return False
        self._cur_mode = mode
        return True



class CUDACallback(Callback):
    def __init__(self, max_steps=50, interval=10, enabled=True):
        self.max_steps = max_steps
        self.interval = interval
        
        self._enabled = enabled

        self._running_epoch_time = []
        self._running_max_memory = []

    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        if not self._enabled:
            return
        # Reset the memory use counter
        # lightning update
        gpu_index = trainer.strategy.root_device.index
        # if int((pl.__version__).split('.')[1])>=7:
        #     gpu_index = trainer.strategy.root_device.index
        # else:
        #     gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._enabled or trainer.global_step >= self.max_steps:
            return
        gpu_index = trainer.strategy.root_device.index
        # if int((pl.__version__).split('.')[1])>=7:
        #     gpu_index = trainer.strategy.root_device.index
        # else:
        #     gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            self._running_epoch_time.append(epoch_time)
            self._running_max_memory.append(max_memory)

            if trainer.global_step % self.interval!= 0:
                epoch_time = sum(self._running_epoch_time) / len(self._running_epoch_time)
                max_memory = max(self._running_max_memory) 
                self._running_epoch_time.clear()
                self._running_epoch_time.clear()
                mainlogger.info(f"[rank {gpu_index}] Average Epoch time: {epoch_time:.2f} seconds")
                mainlogger.info(f"[rank {gpu_index}] Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

class PrintProgressCallback(Callback):

    def __init__(self,
                 interval=100,
                 max_steps=50000,
                 accumulate_grad_batches=1,
                 smooth_coeff: float=0.2,
                 print_test_iterations: bool = False
                 ):
        super().__init__()
        self.interval = interval
        self.max_steps = max_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.smooth_coeff = smooth_coeff
        self.print_test_iterations = print_test_iterations

        self._batch_start_time = None
        self._batch_iter_time = None

        self._is_first_test_batch = True

    def on_train_epoch_start(self, trainer, pl_module):
        mainlogger.info(f"Starting Epoch {trainer.current_epoch + 1}")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_iter_time is None:
            self._batch_iter_time = time.time() - self._batch_start_time
        else:
            self._batch_iter_time = self.smooth_coeff * self._batch_iter_time + (1-self.smooth_coeff) * (time.time() - self._batch_start_time)
        
        if (batch_idx+1) % self.interval == 0:
            eta = (self.max_steps - batch_idx) * self.accumulate_grad_batches * self._batch_iter_time
            formatted_time = format_seconds(eta)
            if 'loss' in outputs:
                loss = outputs["loss"].item()
            mainlogger.info(f"Epoch: {trainer.current_epoch + 1} Batch: {batch_idx + 1}/{len(trainer.train_dataloader)} ETA: {formatted_time}" + (f", Loss: {loss:.4f}" if 'loss' in outputs else ""))

    def on_train_epoch_end(self, trainer, pl_module):
        mainlogger.info(f"Finished Epoch {trainer.current_epoch + 1}")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_iter_time is None:
            self._batch_iter_time = time.time() - self._batch_start_time
        else:
            self._batch_iter_time = self.smooth_coeff * self._batch_iter_time + (1-self.smooth_coeff) * (time.time() - self._batch_start_time)
        eta = (len(trainer.test_dataloaders) - batch_idx - 1) * self.accumulate_grad_batches * self._batch_iter_time
        formatted_time = format_seconds(eta)

        if self._is_first_test_batch:
            self._batch_iter_time = None
        self._is_first_test_batch = False
        mainlogger.info(f"Batch: {batch_idx + 1}/{len(trainer.test_dataloaders)} ETA: {formatted_time}")



class LiveProfiler(Callback):

    def __init__(self, max_steps: int = None, smooth_coeff: float = 0.9, interval: int = 1):
        self.max_steps = max_steps
        self.smooth_coeff = smooth_coeff
        self.interval = interval
        self.iter = 0
        timings = ["forward", "backward", "data_loading", "optim_step", "total"]
        self.timing_dict = {k:None for k in timings}
        self.start_times = {k:None for k in timings}

    def on_before_backward(self, *args, **kwargs):
        self.start_times["backward"] = time.time()
        self._end_cb("forward")
    def on_after_backward(self, *args, **kwargs): self._end_cb("backward")
        

    def on_dataloader_start(self, *args, **kwargs): self.start_times["data_loading"] = time.time()
    def on_dataloader_end(self, *args, **kwargs): self._end_cb("data_loading")

    def on_before_optimizer_step(self, *args, **kwargs): self.start_times["optim_step"] = time.time()
    def on_batch_start(self, *args, **kwargs): self.start_times["total"] = time.time()
    def on_train_batch_end(self, *args, **kwargs): 
        self.start_times["data_loading"] = time.time()

        self._end_cb("optim_step")
        self._end_cb("total")

        if self.iter < self.max_steps and self.iter % self.interval == 0:
            summary_string = self.summary()
            mainlogger.info(summary_string)
        self.iter += 1
    
    def on_train_batch_start(self, *args, **kwargs):
        self.start_times["total"] = time.time()
        self.start_times["forward"] = time.time()

        self._end_cb("data_loading")

    def _end_cb(self, name: str):
        if self.start_times[name] is None:
            return
        if self.timing_dict[name] is None:
            self.timing_dict[name] = time.time() - self.start_times[name]
        else:
            self.timing_dict[name] = self.smooth_coeff*self.timing_dict[name] + (1-self.smooth_coeff)*(time.time() - self.start_times[name])

    def summary(self) -> str:
        output_string = "Timings:  "
        for name, timing in self.timing_dict.items():
            if timing is not None:
                output_string += f"{colored(str(name), 'yellow')}: {timing:.3f}s, "
        return output_string
    

class ModelWatcherCallback(Callback):
    def __init__(self,
                 lower_bound_warn = 1e-7,
                 upper_bound_warn = 1e1,
                 check_inf: bool = True,
                 check_nan: bool = True,
                 max_steps: int = None,
                 log_dir: str = None,
                 log_gradients: bool = False,
                 log_parameters: bool = False,
                 log_loss: bool = False,
                 log_intermediate_values_fwd: bool = False,
                 log_intermediate_values_bwd: bool = False,
                 enabled: bool = True,
                 breakpoints: bool = False,
                 breakpoint_every_n_iterations: int = None,
                 ):
        super().__init__()
        self.lower_bound_warn = lower_bound_warn
        self.upper_bound_warn = upper_bound_warn
        self.check_inf = check_inf
        self.check_nan = check_nan
        self.log_dir = log_dir
        self.breakpoints = breakpoints
        self.breakpoint_every_n_iterations = breakpoint_every_n_iterations
        self._breakpoint = lambda: ipdb.set_trace(); mainlogger.info("Breakpoint reached.")
        self.log_gradients = log_gradients
        self.log_parameters = log_parameters
        self.log_loss = log_loss
        self.log_intermediate_values_fwd = log_intermediate_values_fwd
        self.log_intermediate_values_bwd = log_intermediate_values_bwd

        self._fwd_hooks = {}
        self._bwd_hooks = {}
        self._layerwise_log_dict_fwd = {}
        self._layerwise_log_dict_bwd = {}

        if self.log_gradients:
            os.makedirs(Path(self.log_dir)/"gradients", exist_ok=True)
        if self.log_parameters:
            os.makedirs(Path(self.log_dir)/"parameters", exist_ok=True)
        if self.log_intermediate_values_fwd:
            os.makedirs(Path(self.log_dir)/"intermediate_values"/"forward", exist_ok=True)
        if self.log_intermediate_values_bwd:
            os.makedirs(Path(self.log_dir)/"intermediate_values"/"backward", exist_ok=True)
        #if self.log_loss:
        #    os.makedirs(Path(self.log_dir, "loss"), exist_ok=True)

        self.max_steps = max_steps if max_steps is not None else -1
        self._enabled = enabled
        self._iter = 0
        self._global_step = 0

    def on_before_backward(self, trainer, pl_module, loss):
        if not self._enabled:
            return
        
        # Check if loss is NaN or inf
        
        if not torch.isfinite(loss):
            mainlogger.error(f"[rank {trainer.global_rank}] NaN or inf value detected in loss before backward: {loss}")
        if self.log_intermediate_values_bwd:
            mainlogger.info("Checking model gradients...")

        if self.log_loss:
            self._log_loss(trainer, pl_module, loss, trainer.global_rank)
        
        if self.breakpoints: self._breakpoint()


    def on_after_backward(self, trainer, pl_module):
        if not self._enabled:
            return
        # Check gradients
        if not self.log_intermediate_values_bwd:
            mainlogger.info("Checking model gradients...")
        success = True
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is None:
                mainlogger.warning(f"[rank {trainer.global_rank}] {name}: Requires gradient but not gradient was not computed.")
                success = False
                continue
            if param.grad is not None:
                grad = param.grad
                if self.check_inf:
                    is_inf = torch.isinf(grad)
                    if is_inf.any():
                        mainlogger.warning(f"{name}: Gradient contains {torch.sum(is_inf)}/{grad.numel()} inf values.")
                        success = False
                if self.check_nan:
                    is_nan = torch.isnan(grad)
                    if is_nan.any():
                        mainlogger.warning(f"[rank {trainer.global_rank}] {name}: Gradient contains {torch.sum(is_nan)}/{grad.numel()} NaN values.")
                        success = False
                under_lower_bound = torch.abs(grad) < self.lower_bound_warn
                if under_lower_bound.any():
                    mainlogger.warning(f"[rank {trainer.global_rank}] {name}: Gradient has {torch.sum(under_lower_bound)}/{grad.numel()} elements under lower bound")
                    success = False
                
                over_upper_bound = torch.abs(grad) > self.upper_bound_warn
                if over_upper_bound.any():
                    mainlogger.warning(f"[rank {trainer.global_rank}] {name}: Gradient has {torch.sum(over_upper_bound)}/{grad.numel()} elements over upper bound")
                    success = False
        if success:
            mainlogger.info(f"[rank {trainer.global_rank}] All model gradients are valid!")
        else:
            mainlogger.error(f"[rank {trainer.global_rank}] Invalid model gradients detected!")
        if self.log_gradients and not self.log_intermediate_values_bwd:
            self._log_gradients(pl_module, trainer.global_rank)
        else:
            # Handle gradient saving through the registered backward hooks
            if self.log_intermediate_values_bwd:
                outfile = Path(self.log_dir) / "intermediate_values" / "backward" / f"step_{pl_module.global_step}_rank_{trainer.global_rank}.pth"
                with open(outfile, 'wb') as f:
                    torch.save(self._layerwise_log_dict_bwd, f)
                self._layerwise_log_dict_bwd = {}
        if self.breakpoints: self._breakpoint()
        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check model outputs
        if not self._enabled:
            return
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
            if not torch.isfinite(loss):
                mainlogger.warning(f"[rank {trainer.global_rank}] NaN or inf value detected in loss: {loss}")
        
        # Check other outputs if available
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and key != 'loss':
                if not torch.isfinite(value).all():
                    mainlogger.warning(f"[rank {trainer.global_rank}] Output '{key}': NaN or inf value detected: {value}")

        if self.log_intermediate_values_fwd:
            outfile = Path(self.log_dir) / "intermediate_values" / "forward" / f"step_{pl_module.global_step}_idx_{batch_idx}_rank_{trainer.global_rank}.pth"
            with open(outfile, 'wb') as f:
                torch.save(self._layerwise_log_dict_fwd, f)
            self._layerwise_log_dict_fwd = {}
        if self.breakpoint_every_n_iterations is not None:
            if (pl_module.global_step+1) % self.breakpoint_every_n_iterations == 0:
                self._breakpoint()
        if self.breakpoints: self._breakpoint()
        

    def on_sanity_check_start(self, trainer, pl_module):
        if not self._enabled:
            return
        self._check_model_parameters(pl_module)
        if self.breakpoints: self._breakpoint()


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._enabled:
            return
        
        self._iter += 1
        self._global_step = pl_module.global_step
        if self.max_steps > 0 and self._iter > self.max_steps:
            mainlogger.info(f"[rank {trainer.global_rank}] Model watcher: Maximum iterations reached")
            self._remove_hooks(trainer.global_rank)
            self._enabled = False
        mainlogger.info(f"[rank {trainer.global_rank}] Model watcher iteration: {pl_module.global_step}/{self.max_steps}")
        self._check_model_parameters(pl_module)
        if self.log_parameters:
            self._log_parameters(pl_module, trainer.global_rank)
        if self.breakpoints: self._breakpoint()
        

    def on_fit_start(self, trainer, pl_module):
        # Register hooks on specified layers
        if self.log_intermediate_values_fwd or self.log_intermediate_values_bwd:
            for name, module in pl_module.named_modules():
                if name in ['']:
                    continue
                req_params=  [True for param in module.parameters() if param.requires_grad]
                if len(req_params) > 0:
                    if self.log_intermediate_values_fwd:
                        hook = module.register_forward_hook(self._hook_fn(name, forward=True, global_rank=trainer.global_rank))
                        self._fwd_hooks[name] = hook
                    if self.log_intermediate_values_bwd:
                        hook = module.register_full_backward_hook(self._hook_fn(name, forward=False, global_rank=trainer.global_rank))
                        self._bwd_hooks[name] = hook
                    mainlogger.info(f"[rank {trainer.global_rank}] Registered hooks on {name}")
        if self.breakpoints: self._breakpoint()

    def on_fit_end(self, trainer, pl_module):
        self._remove_hooks()
        if self.breakpoints: self._breakpoint()


    def _remove_hooks(self, global_rank: int = 0):
        for k, v in self._fwd_hooks.items(): 
            v.remove()
            mainlogger.info(f"[rank {global_rank}] Removed forward hook from {k}")
        for k, v in self._bwd_hooks.items():
            v.remove()
            mainlogger.info(f"[rank {global_rank}] Removed backward hook from {k}")


    def _hook_fn(self, name, forward:bool = True, global_rank:int = 0):

        def hook_impl_fwd(module, input, output):
  
            if isinstance(output, torch.Tensor):
                output_ = (output.detach().cpu(),)
            else:
                output_ = move_tensors_to_cpu(output)

            input_not_finite = not torch.isfinite(input[0].detach().cpu()).all()
            output_not_finite = not torch.isfinite(output_[0]).all()


            input_save = move_tensors_to_cpu(input)
            output_save = output_
            self._layerwise_log_dict_fwd[name] = (input_save, output_save)

            if output_not_finite and not input_not_finite:
                mainlogger.warning(f"[rank {global_rank}] {name}: Input is not finite!")
            elif input_not_finite and not output_not_finite:
                mainlogger.warning(f"[rank {global_rank}] {name}: Output is not finite!")
            elif input_not_finite and output_not_finite:
                mainlogger.warning(f"[rank {global_rank}] {name}: Input and output are not finite!")


        def hook_impl_bwd(module, grad_input, grad_output):
            grad_input_ = None
            if grad_input is not None and (not isinstance(grad_input, torch.Tensor) and len(grad_input) > 0):
                if isinstance(grad_input, torch.Tensor): grad_input_ = (grad_input,)
                else: grad_input_ = grad_input

                grad_input_not_finite = not torch.isfinite(grad_input_[0].detach().cpu()).all() if grad_input_[0] is not None else False

            grad_output_not_finite = not torch.isfinite(grad_output[0].detach().cpu()).all()

            input_save = move_tensors_to_cpu(grad_input_) if grad_input_ is not None else None
            output_save = move_tensors_to_cpu(grad_output) if grad_output is not None else None
            self._layerwise_log_dict_bwd[name] = (input_save, output_save)

            if grad_output_not_finite and not grad_input_not_finite:
                mainlogger.warning(f"[rank {global_rank}] {name}: Gradient input is not finite!")
            elif grad_input_not_finite and not grad_output_not_finite:
                mainlogger.warning(f"[rank {global_rank}] {name}: Gradient is not finite!")
            elif grad_input_not_finite and grad_output_not_finite:
                mainlogger.warning(f"[rank {global_rank}] {name}: Gradient Input and gradient itself are not finite!")


        if forward: return hook_impl_fwd
        else: return hook_impl_bwd

    def _check_model_parameters(self, pl_module, global_rank: int = 0):
        mainlogger.info("Checking model parameters...")
        success = True
        for name, param in pl_module.named_parameters():
            if self.check_inf:
                is_inf = torch.isinf(param)
                if is_inf.any():
                    mainlogger.warning(f"[rank {global_rank}] {name}: Parameter contains {torch.sum(is_inf)}//{param.numel()} inf values.")
                    success = False
            if self.check_nan:
                is_nan = torch.isnan(param)
                if is_nan.any():
                    mainlogger.warning(f"[rank {global_rank}] {name}: Parameter contains {torch.sum(is_nan)}/{param.numel()} NaN values.")
                    success = False
            under_lower_bound = torch.abs(param) < self.lower_bound_warn
            total_under = torch.sum(under_lower_bound)
            if total_under > 0.3 * param.numel():
                mainlogger.warning(f"[rank {global_rank}] {name}: Parameter has {torch.sum(under_lower_bound)}/{param.numel()} elements under lower bound")
                success = False
            
            over_upper_bound = torch.abs(param) > self.upper_bound_warn
            total_over = torch.sum(over_upper_bound)
            if total_over > 0.3 * param.numel():
                mainlogger.warning(f"[rank {global_rank}] {name}: Parameter has {torch.sum(over_upper_bound)}/{param.numel()} elements over upper bound")
                success = False
        if success:
            mainlogger.info("[rank {global_rank}] All model parameters are valid!")

    def _log_parameters(self, pl_module, only_trainable: bool = True, global_rank: int = 0):
        save_dict = {}
        for name, param in pl_module.named_parameters():
            if only_trainable and not param.requires_grad:
                continue
            save_dict[name] = param.detach().cpu()
        with open(Path(self.log_dir) / "parameters" / f"step_{pl_module.global_step}_rank_{global_rank}.pth", "wb") as f:
            torch.save(save_dict, f)

    def _log_gradients(self, pl_module, global_rank: int = 0):
        save_dict = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                save_dict[name] = param.grad.detach().cpu()
        with open(Path(self.log_dir) / "gradients" / f"step_{pl_module.global_step}_rank_{global_rank}.pth", "wb") as f:
            torch.save(save_dict, f)

    def _log_loss(self, trainer, pl_module, loss, global_rank: int = 0):

        flag = "w" if not os.path.exists(Path(self.log_dir) / f"loss_rank_{global_rank}.txt") else 'a'
        with open(Path(self.log_dir) / f"loss_rank_{global_rank}.txt", flag) as f:
            f.write(f"{trainer.global_step}: {loss.detach().cpu().item()}\n")

    


class FreezeCallback(Callback):

    def __init__(self, parameters: List[str], freeze_steps: int):

        self.parameters = parameters
        self.freeze_steps = freeze_steps

        self._is_frozen = True

    def on_fit_start(self, trainer, pl_module):
        
        state_dict = pl_module.state_dict()
        for i, param in enumerate(self.parameters):
            if param not in state_dict:
                mainlogger.warning(f"Parameter '{param}' not found in the model. FreezeCallback will exclude it.")
                del self.parameters[i]
        
        if self.freeze_steps == 0:
            self._unfreeze(pl_module)
        else:
            self._freeze(pl_module)

    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        if self._is_frozen and (trainer.global_step % self.freeze_steps-1 == 0):
            self._unfreeze(pl_module)

    def _unfreeze(self, pl_module):
        for param in self.parameters:
            pl_module.state_dict()[param].requires_grad = True
        self._is_frozen = False
        mainlogger.info("Unfreezing parameters...")
    
    def _freeze(self, pl_module):
        for param in self.parameters:
            pl_module.state_dict()[param].requires_grad = False
        self._is_frozen = True
        mainlogger.info("Freezing parameters...")