"""
    Evaluation Script 

    
"""

import sys
import os
import argparse
import yaml
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List  # Added List for type annotations
from termcolor import colored
import time
from datetime import datetime
import pandas as pd
import traceback

from einops import rearrange

from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

from utils.evaluation import *

DEFAULT_EVALUATION_FILE = "results/evaluation.csv"

###########################################################
#################### FVD Computation ######################
###########################################################

def get_video_path_pairs(path: str):
    """
    Generate pairs of ground truth and generated video file paths from directories.

    This function iterates over the subdirectories in the given path and yields tuples 
    containing the paths for the ground truth video ("ground_truth.mp4") and the generated 
    video ("generated.mp4") if both exist.

    Args:
        path (str): The path to the directory containing video subdirectories.

    Yields:
        tuple[str, str]: A tuple containing the ground truth video path and the generated video path.
    """
    path = Path(path)

    video_dirs = sorted([p for p in path.iterdir() if p.is_dir()])
    for dir in video_dirs:
        gt_file = dir / "ground_truth.mp4"
        gen_file = dir / "generated.mp4"

        if gt_file.exists() and gen_file.exists():
            yield (str(gt_file), str(gen_file))
        else:
            print(f"Missing ground truth or generated video for {dir}")

def load_videos(paths, desc):
    """
    Load videos from the provided file paths and stack them into a tensor.

    This function uses the 'load_video' method from fvdcal.video_preprocess to load each video,
    then stacks them into a single tensor.

    Args:
        paths (iterable): An iterable of video file paths.
        desc (str): A description string (unused in this implementation).

    Returns:
        torch.Tensor: A tensor containing all loaded videos.
    """
    from fvdcal.video_preprocess import load_video
    return torch.stack([load_video(path, num_frames=None) for path in paths])

def fvd(
        path, 
        output, 
        max_videos_per_batch: int = None, 
        max_videos: int = None,
        sample_list: str = None,
        model_path: str = "ckpts"
        ):
    """
    Compute the FVD (Fréchet Video Distance) scores for videos in a directory.

    This function calculates FVD scores using two different methods ('videogpt' and 'stylegan') 
    on pairs of ground truth and generated videos found in the specified path. It optionally 
    subsamples the videos based on a provided sample list or maximum video limit.

    Args:
        path (str): Path to the directory containing video subdirectories.
        output (str): Directory where the evaluation results will be saved.
        max_videos_per_batch (int, optional): Maximum number of videos to process per batch.
        max_videos (int, optional): Maximum number of videos to evaluate.
        sample_list (str, optional): Path to a file containing a list of samples to include.
        model_path (str, optional): Path to the model directory. Defaults to "ckpts".

    Returns:
        tuple[float, float]: FVD scores computed using the 'videogpt' and 'stylegan' methods.
    """
    from fvdcal import FVDCalculation, FVDCalculation2
    print(colored("Starting FVD evaluation...", "green"))

    fvd_videogpt_calculator = FVDCalculation2(method="videogpt", batch_size=max_videos_per_batch)
    fvd_stylegan_calculator = FVDCalculation2(method="stylegan", batch_size=max_videos_per_batch)
    #model_path = "src/evaluation/FVD/model"

    video_list = list(get_video_path_pairs(path))
    if sample_list is not None:
        with open(sample_list, "r") as f:
            valid_sample_list = [line.strip() for line in f.readlines()]
        video_list_subsample = []
        for gt_path, gen_path in video_list:
            if str(Path(gt_path).parent.stem) in valid_sample_list:
                video_list_subsample.append((gt_path, gen_path))
        video_list = video_list_subsample
    if max_videos is not None:
        video_list = video_list[:min(len(video_list), max_videos)]

    video_list_gt = [v[0] for v in video_list]
    video_list_gen = [v[1] for v in video_list]
    
    print(f"Found {len(video_list)} videos to evaluate!")

    fvd_videogpt = fvd_videogpt_calculator(video_list_gt, video_list_gen, model_path=model_path)
    fvd_stylegan = fvd_stylegan_calculator(video_list_gt, video_list_gen, model_path=model_path)

    print(colored("FVD evaluation finished!", "green"))

    save_dict = {
        "image_path": path,
        "num_videos": len(video_list),
        "output_path": output,
        "fvd_videogpt": fvd_videogpt.item(),
        "fvd_stylegan": fvd_stylegan.item(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    out_file = Path(output) / "fvd_evaluation.yaml"
    with open(out_file, "w") as file:
        yaml.safe_dump(save_dict, file, default_flow_style=False, sort_keys=False)

    return fvd_videogpt.item(), fvd_stylegan.item()

###########################################################
################ Camera Pose Evaluation ###################
###########################################################

def compute_camera_poses(img_dir: str, pose_dir: str, f: float, cx: float, cy: float, use_colmap=False, suppress_output: bool = False) -> tuple:
    """
    Compute relative camera poses from video frames using COLMAP or GLOMAP.

    This function extracts frames from a video and computes camera poses by running
    feature extraction, matching, and mapping commands. It then converts the results
    to obtain the relative camera-to-world transformation.

    Args:
        img_dir (str): Directory where extracted video frames are stored.
        pose_dir (str): Directory to store pose estimation results.
        f (float): Focal length.
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.
        use_colmap (bool, optional): Whether to use COLMAP instead of GLOMAP. Defaults to False.
        suppress_output (bool, optional): If True, suppresses command output messages. Defaults to False.

    Returns:
        tuple: The computed relative camera pose transformation.
    """
    def convert(config: dict) -> list[str]:
        return sum([[f"--{k}", f"{v}"] for k, v in config.items()], [])
    
    def run_command(cmd):
        if not suppress_output:
            print(colored("Running: ", "yellow") + " ".join(cmd))
        result = subprocess.run(cmd, shell=False)
        if result.returncode != 0:
            error_message = result.stderr.decode("utf-8")
            print(colored("Error:", "red"), error_message)
            return False
        return True

    model_dir = f"{pose_dir}/model"
    os.makedirs(model_dir, exist_ok=True)

    db_path = f"{pose_dir}/database.db"

    if os.path.exists(db_path):
        os.remove(db_path)

    config = {
        "feature_extractor": {
            "database_path": db_path,
            "image_path": img_dir,
            "ImageReader.single_camera": 1,
            "ImageReader.camera_model": "SIMPLE_PINHOLE",
            "ImageReader.camera_params": f"{f},{cx},{cy}",
            "SiftExtraction.estimate_affine_shape": 1,
            "SiftExtraction.domain_size_pooling": 1,
        },
        "sequential_matcher": {
            "database_path": db_path,
            "SiftMatching.guided_matching": 1,
            "SiftMatching.max_num_matches": 65536,
        },
        "mapper": {
            "database_path": db_path,
            "image_path": img_dir,
            "output_path": model_dir,
            "output_format": "txt",
            "RelPoseEstimation.max_epipolar_error": 4,
            "BundleAdjustment.optimize_intrinsics": 0,
        },
    }

    if not run_command(["colmap", "feature_extractor"] + convert(config["feature_extractor"])):
        return None
    if not run_command(["colmap", "sequential_matcher"] + convert(config["sequential_matcher"])):
        return None
    if not run_command(["glomap" if not use_colmap else "colmap", "mapper"] + convert(config["mapper"])):
        return None

    write_depth_pose_from_colmap_format(f"{model_dir}/0", model_dir, ext=".txt")

    w2c = rt34_to_44(get_rt(f"{model_dir}/poses"))
    c2w = w2c.inverse()
    rel_c2w = relative_pose(c2w, mode="left")

    return rel_c2w


def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:  # N, 3, 3
    """
    Calculate the rotation error between two rotation matrices.

    Args:
        r1 (Tensor): First rotation matrix tensor of shape (N, 3, 3).
        r2 (Tensor): Second rotation matrix tensor of shape (N, 3, 3).

    Returns:
        Tensor: The rotation error in radians.
    """
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:  # N, 3
    """
    Calculate the translation error between two translation vectors.

    Args:
        t1 (Tensor): First translation vector tensor of shape (N, 3).
        t2 (Tensor): Second translation vector tensor of shape (N, 3).

    Returns:
        Tensor: The Euclidean distance (L2 norm) between the two translation vectors.
    """
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:  # N, 3, 4
    """
    Calculate the camera metric error between two camera pose representations.

    Args:
        rt1 (Tensor): First camera pose tensor of shape (N, 3, 4).
        rt2 (Tensor): Second camera pose tensor of shape (N, 3, 4).

    Returns:
        Tensor: The computed camera metric error.
    """
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)

def metric(c2w_1: Tensor, c2w_2: Tensor) -> tuple[float, float, float]:
    """
    Compute the overall error metrics between two sets of camera poses.

    This function computes the total rotation error, translation error, and camera metric error 
    between the provided camera-to-world transformation tensors.

    Args:
        c2w_1 (Tensor): First set of camera-to-world transformations of shape (N, 3, 4).
        c2w_2 (Tensor): Second set of camera-to-world transformations of shape (N, 3, 4).

    Returns:
        tuple[float, float, float]: A tuple containing the rotation error, translation error, and camera metric error.
    """
    RotErr = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).sum().item()

    c2w_1_rel = normalize_t(c2w_1, c2w_1)
    c2w_2_rel = normalize_t(c2w_2, c2w_2)

    TransErr = calc_transerr(c2w_1_rel[:, :3, 3], c2w_2_rel[:, :3, 3]).sum().item()
    CamMC = calc_cammc(c2w_1_rel[:, :3, :4], c2w_2_rel[:, :3, :4]).sum().item()

    return RotErr, TransErr, CamMC


def camera_pose_evaluation(path,
                            output,
                            max_videos: int = None,
                            use_colmap=False,
                            trials_per_video: int = 1,
                            trial_strategy: "Literal['average', 'best']" = "average",
                            sort_videos: bool = False,
                            sample_list: str = None,
                            ) -> Tuple[float, float, float]:
    """
    Evaluate camera pose estimation errors for generated videos.

    This function computes camera pose errors by comparing estimated poses with ground truth 
    poses for each video directory in the specified path. It supports multiple trials per video 
    and aggregates the errors using either an average or best strategy.

    Args:
        path (str): Path to the directory containing video subdirectories.
        output (str): Directory where temporary and evaluation outputs will be saved.
        max_videos (int, optional): Maximum number of video directories to evaluate.
        use_colmap (bool, optional): Whether to use COLMAP for pose estimation. Defaults to False.
        trials_per_video (int, optional): Number of trials to perform per video. Defaults to 1.
        trial_strategy (str, optional): Strategy to aggregate trials ("average" or "best"). Defaults to "average".
        sort_videos (bool, optional): If True, sorts the video directories. Defaults to False.
        sample_list (str, optional): Path to a file containing a list of samples to include.

    Returns:
        Tuple[float, float, float]: Average rotation error, translation error, and camera metric error.
    """
    print("Starting camera pose evaluation...")
    eval_paths = [p for p in Path(path).iterdir() if p.is_dir()]
    if sort_videos:
        eval_paths = sorted(eval_paths, key=lambda x: str(Path(x).stem))

    tmp_dir = Path(output) / "tmp"
    detail_eval_file = Path(output) / "camera_eval.yaml"
    if sample_list is not None:
        with open(sample_list, "r") as f:
            valid_sample_list = [line.strip() for line in f.readlines()]
        video_list_subsample = []
        for p in eval_paths:
            if str(Path(p).stem) in valid_sample_list:
                video_list_subsample.append(p)
        eval_paths = video_list_subsample

    eval_paths = eval_paths[:min(len(eval_paths), max_videos)] if max_videos else eval_paths
    print(f"Found {len(eval_paths)} videos to evaluate!")
    
    os.makedirs(tmp_dir, exist_ok=True)

    save_dict = {}
    rot_err_list = []
    trans_err_list = []
    cam_mc_list = []
    for i, p in tqdm(enumerate(eval_paths), total=len(eval_paths)):
        cam_data_file = p / "camera_data.npy"
        video_file = p / "generated.mp4"
        if not cam_data_file.exists():
            print(f"Could not find camera data for {str(p.stem)}. Skipping...")
            continue
        try:
            cam_data_gt = torch.from_numpy(np.load(cam_data_file)).float()
            cam_data_gt = cam_data_gt[:, 1:] 
            gt_w2c = cam_data_gt[:, 6:].reshape((-1, 3, 4))
            gt_c2w = rt34_to_44(gt_w2c).inverse()
            gt_rel_c2w = relative_pose(gt_c2w, mode="left")

            img_dir = f"{tmp_dir}/img"
            os.makedirs(img_dir, exist_ok=True)
            get_frames(str(video_file), img_dir)

            start = time.perf_counter()

            fx, fy, cx, cy = cam_data_gt[0, :4]
            trial_rot_err = []
            trial_trans_err = []
            trial_cam_mc = []
            for _ in range(trials_per_video):
                sample_rel_c2w = compute_camera_poses(img_dir, f"{tmp_dir}/pose", fx, cx, cy, use_colmap=use_colmap, suppress_output=True)
                if sample_rel_c2w is None:
                    continue
                num_gen_imgs = sample_rel_c2w.shape[0]
                gt_rel_c2w = relative_pose(gt_c2w[:num_gen_imgs], mode="left")
                rot_err, trans_err, cam_mc = metric(gt_rel_c2w.float().clone(), sample_rel_c2w.float().clone())
                trial_rot_err.append(rot_err)
                trial_trans_err.append(trans_err)
                trial_cam_mc.append(cam_mc)
            if len(trial_cam_mc) == 0:
                continue
            elif len(trial_cam_mc) == 1:
                rot_err, trans_err, cam_mc = trial_rot_err[0], trial_trans_err[0], trial_cam_mc[0]
            else:
                if trial_strategy == "average":
                    rot_err, trans_err, cam_mc = np.mean(trial_rot_err), np.mean(trial_trans_err), np.mean(trial_cam_mc)
                elif trial_strategy == "best":
                    min_index = np.argmin(cam_mc)
                    rot_err, trans_err, cam_mc = trial_rot_err[min_index], trial_trans_err[min_index], trial_cam_mc[min_index]
                else:
                    raise ValueError(f"Invalid trial strategy: {trial_strategy}")

            end = time.perf_counter()

            save_dict[str(p.stem)] = {
                "RotErr": rot_err,
                "TransErr": trans_err,
                "CamMC": cam_mc,
                "Time": end - start
            }
            rot_err_list.append(rot_err)
            trans_err_list.append(trans_err)
            cam_mc_list.append(cam_mc)
        except Exception as e:
            print(colored(f"Error processing '{str(p.stem)}': {str(e)}", "red"))
            traceback.print_exc()

    with open(detail_eval_file, "w") as file:
        yaml.dump(save_dict, file, default_flow_style=False, sort_keys=False)
    print(colored(f"Camera pose evaluation finished!", "green"))

    return np.mean(rot_err_list), np.mean(trans_err_list), np.mean(cam_mc_list)

def compute_extended_metrics(path, output, max_videos_per_batch=None, max_videos=None, sample_list=None):
    """
    Compute extended evaluation metrics for videos.

    This function calculates metrics such as MSE, RMSE, SSIM, and LPIPS for a set of video pairs.
    Videos are loaded in batches and evaluated frame-by-frame, then the results are aggregated.

    Args:
        path (str): Path to the directory containing video subdirectories.
        output (str): Directory where detailed evaluation results will be saved.
        max_videos_per_batch (int, optional): Maximum number of videos to process per batch.
        max_videos (int, optional): Maximum number of videos to evaluate.
        sample_list (str, optional): Path to a file containing a list of samples to include.

    Returns:
        tuple: A tuple containing the total MSE, RMSE, average SSIM, and average LPIPS.
    """
    video_list = list(get_video_path_pairs(path))
    if sample_list is not None:
        with open(sample_list, "r") as f:
            valid_sample_list = [line.strip() for line in f.readlines()]
        video_list_subsample = []
        for gt_path, gen_path in video_list:
            if str(Path(gt_path).parent.stem) in valid_sample_list:
                video_list_subsample.append((gt_path, gen_path))
        video_list = video_list_subsample
    if max_videos:
        video_list = video_list[:min(len(video_list), max_videos)]
    print(f"Found {len(video_list)} videos to evaluate!")
    if max_videos_per_batch is not None and len(video_list) > max_videos_per_batch:
        video_list = [video_list[i:min(len(video_list), i + max_videos_per_batch)] for i in range(0, len(video_list), max_videos_per_batch)]
    else:
        video_list = [video_list]

    ssim_eval = StructuralSimilarityIndexMeasure(data_range=255)
    lpips_eval = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    mse_total_list = []
    mse_per_timestep_list = []
    ssim_list = []
    lpips_list = []
    ssim_per_timestep_list = []
    lpips_per_timestep_list = []

    for i, batch in tqdm(enumerate(video_list), total=len(video_list)):
        video_gt_paths = [video[0] for video in batch]
        video_gen_paths = [video[1] for video in batch]

        gt_videos = load_videos(video_gt_paths, "loading real videos")
        sample_videos = load_videos(video_gen_paths, "loading generated videos")

        mse_total = torch.mean((gt_videos - sample_videos) ** 2)
        mse_per_timestep = torch.mean((gt_videos - sample_videos) ** 2, dim=[i for i in range(sample_videos.dim()) if i != 1])

        lpips_timestep_list = []
        ssim_timestep_list = []
        for t in range(sample_videos.shape[1]):
            ssim_timestep_list.append(ssim_eval(sample_videos[:, t], gt_videos[:, t]))
            lpips_timestep_list.append(lpips_eval(sample_videos[:, t], gt_videos[:, t]))

        ssim = np.mean(ssim_timestep_list)
        lpips = np.mean(lpips_timestep_list)

        mse_total_list.append(mse_total.item())
        mse_per_timestep_list.append(mse_per_timestep)

        ssim_list.append(ssim)
        lpips_list.append(lpips)
        ssim_per_timestep_list.append(ssim_timestep_list)
        lpips_per_timestep_list.append(lpips_timestep_list)

    mse_total = float(np.mean(mse_total_list))
    rmse = float(np.sqrt(mse_total))
    mse_per_timestep = torch.mean(torch.stack(mse_per_timestep_list, dim=0), dim=0).tolist()

    ssim = float(np.mean(ssim_list))
    lpips = float(np.mean(lpips_list))
    ssim_per_timestep = np.mean(np.stack(ssim_per_timestep_list, axis=0), axis=0).tolist()
    lpips_per_timestep = np.mean(np.stack(lpips_per_timestep_list, axis=0), axis=0).tolist()

    detail_eval_file = Path(output) / "frame_eval.yaml"
    save_dict = {
        "mse_total": mse_total,
        "mse_per_timestep": mse_per_timestep,
        "rmse": rmse,
        "ssim": ssim,
        "lpips": lpips,
        "ssim_per_timestep": ssim_per_timestep,
        "lpips_per_timestep": lpips_per_timestep
    }
    os.makedirs(detail_eval_file.parent, exist_ok=True)
    with open(detail_eval_file, "w") as file:
        yaml.dump(save_dict, file, default_flow_style=False, sort_keys=False)

    return mse_total, rmse, ssim, lpips

###########################################################
########################## Main ###########################
###########################################################

def arguments():
    """
    Parse command-line arguments for the evaluation script.

    This function sets up an argument parser to handle various evaluation options such as 
    input/output paths, which metrics to compute (FVD, extended metrics, camera pose evaluation),
    and other configuration parameters.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to manage experiments")
    parser.add_argument("-p", "--path", type=str, help="Path to image directories")
    parser.add_argument("--max-videos-in-mem", type=int, default=None, help="Maximum number of videos to load into memory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("--fvd", action="store_true", default=False, help="Compute FVD scores")
    parser.add_argument("--extended", action="store_true", default=False, help="Compute extended evaluation scores.")
    parser.add_argument("--glomap", action="store_true", default=False, help="Compute GLOMAP metrics on camera poses")
    parser.add_argument("--colmap", action="store_true", default=False, help="Compute GLOMAP metrics on camera poses")
    parser.add_argument("-n", "--name", type=str, default=None, help="Trial name")
    parser.add_argument("--evaluation-file", type=str, default=DEFAULT_EVALUATION_FILE, help="Evaluation file path")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to evaluate")
    parser.add_argument("--sample-list", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=1)

    args = parser.parse_args()
    return args

def main(args):
    """
    Main function to perform evaluation based on provided arguments.

    This function handles the evaluation process by computing various metrics (FVD, camera pose errors,
    extended metrics) depending on the command-line options provided. It saves the results to files and
    prints a summary of the evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    fvd_videogpt, fvd_stylegan, rot_err, trans_err, cam_mc = -1, -1, -1, -1, -1
    mse, ssim, lpips, rmse = -1, -1, -1, -1

    if args.fvd:
        fvd_stylegan, fvd_videogpt = fvd(args.path, args.output, args.max_videos_in_mem, args.max_videos, args.sample_list)

    if args.colmap or args.glomap:
        rot_err, trans_err, cam_mc = camera_pose_evaluation(
            path=args.path,
            output=args.output,
            max_videos=args.max_videos,
            trials_per_video=5,
            sample_list=args.sample_list
        )
    
    if args.extended:
        mse, rmse, ssim, lpips = compute_extended_metrics(args.path, args.output, args.max_videos_in_mem, args.max_videos, args.sample_list)

    eval_dict = {
        "Run name": args.name if args.name is not None else "N/A",
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "FVD (StyleGAN)": fvd_stylegan,
        "FVD (VideoGPT)": fvd_videogpt,
        "RotErr": rot_err,
        "TransErr": trans_err,
        "CamMC": cam_mc,
        "Input path": str(Path(args.path).resolve()),
        "Output path": str(Path(args.output).resolve()),
    }

    print(f"Savin results to: {args.evaluation_file}")
    if os.path.exists(args.evaluation_file):
        eval_data = pd.read_csv(args.evaluation_file)
        eval_data = eval_data._append(eval_dict, ignore_index=True).sort_values(by="Time", ascending=False)
    else:
        eval_data = pd.DataFrame([eval_dict])
    
    eval_data.to_csv(args.evaluation_file, index=False)

    print("Results:")
    print(f"  FVD (StyleGAN): {fvd_stylegan:.4f}")
    print(f"  FVD (VideoGPT): {fvd_videogpt:.4f}")
    print(f"  RotErr:         {rot_err:.4f}")
    print(f"  TransErr:       {trans_err:.4f}")
    print(f"  CamMC:          {cam_mc:.4f}")
    print(f"  MSE:            {mse:.4f}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  SSIM:           {ssim:.4f}")
    print(f"  LPIPS:          {lpips:.4f}")
    

if __name__ == "__main__":
    args = arguments()
    main(args)
