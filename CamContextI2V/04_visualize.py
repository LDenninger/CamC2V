import gradio as gr
import json
import argparse
import cv2
import socket
import numpy as np
import torch
import torchvision
import os
from einops import rearrange
from pathlib import Path

from utils.utils import instantiate_from_config
from main.runtime import Image2Video, bezier_curve

# Parse command-line arguments for the demo application.
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--result_dir", type=str, default="./results/demo")
parser.add_argument("--data_dir", type=str, default="/home/group-cvg/datasets/realestate10k_new")
parser.add_argument("--model_meta_path", type=str, default="./configs/demo/models.json")
parser.add_argument("--example_meta_path", type=str, default="")
parser.add_argument("--camera_pose_meta_path", type=str, default="./configs/demo/camera_poses.json")
parser.add_argument("--experiment-path", type=str, default="/home/denninge/CamContextI2V/results")
parser.add_argument("--use_qwen2vl_captioner", action="store_true")
parser.add_argument("--use_host_ip", action="store_true")
parser.add_argument("--no-overlay-indices", action="store_false", default=True)
parser.add_argument("--detect-models", action="store_true", default=False)
parser.add_argument("--frame-stride", type=int, default=8)
parser.add_argument("--sample-file", type=str, default="/home/group-cvg/datasets/realestate10k_new/test_valid_list.txt")
args = parser.parse_args()

def load_models():
    """
    Load model metadata from a JSON file and filter out models with 'interp' in their names.

    Returns:
        tuple: A tuple containing a list of model names (filtered) and the complete model metadata dictionary.
    """
    with open(args.model_meta_path, "r") as f:
        data = json.load(f)
    return list(filter(lambda x: "interp" not in x, data.keys())), data

def detect_models():
    """
    Detect available models from the experiment directory by scanning checkpoint subdirectories.

    Returns:
        tuple: A tuple containing a list of detected model names and a dictionary mapping model names 
               to their metadata (including config file path, checkpoint path, width, and height).
    """
    experiments = [p for p in Path(args.experiment_path).iterdir() if p.is_dir()]
    checkpoint_sub_dir = Path("checkpoints", "trainstep_checkpoints")

    model_dictionary = {}
    model_names = []
    for epath in experiments:
        ckpt_dir = epath / checkpoint_sub_dir
        if not os.path.exists(ckpt_dir):
            continue
        for ckpt in ckpt_dir.iterdir():
            if not ckpt.is_dir():
                continue
            ckpt_path = ckpt / "checkpoint" / "mp_rank_00_model_states.pt"
            config_path = epath / "config.yaml"
            model_name = "_".join(str(epath.stem).split("_")[:-2]) + f"_{ckpt.stem}"
            model_dictionary[model_name] = {
                "config_file": config_path,
                "ckpt_path": ckpt_path,
                "width": 256,
                "height": 256
            }
            model_names.append(model_name)
    return model_names, model_dictionary

def load_camera_pose_type():
    """
    Load available camera pose types from a JSON metadata file.

    Returns:
        list: A list of camera pose types, including "original" and additional types from the file.
    """
    with open(args.camera_pose_meta_path, "r") as f:
        data = json.load(f)
    pose_types = ["original"] + list(data.keys())
    return pose_types

def load_dataset():
    """
    Instantiate and load the dataset using a configuration dictionary.

    Returns:
        Dataset: The instantiated dataset object.
    """
    data_config = {
        "target": "data.realestate10k.RealEstate10K",
        "params":
            {
                "data_dir":  f"{args.data_dir}/video_clips/test",
                "meta_path": f"{args.data_dir}/valid_metadata/test",
                "meta_list": args.sample_file,
                "caption_file": f"{args.data_dir}/test_captions.json",
                "video_length": -1,
                "frame_stride": args.frame_stride,
                "frame_stride_for_condition": 0,
                "resolution": [256, 256],
                "spatial_transform": "resize_center_crop",
                "invert_video": False,
                "return_full_clip": False,                
            }
    }
    dataset = instantiate_from_config(data_config)
    return dataset

def load_video_names():
    """
    Placeholder function for loading video names.

    Returns:
        list: An empty list.
    """
    return []

def overlay_indices(frames):
    """
    Overlay frame indices on each frame image.

    Args:
        frames (list): List of frame images as numpy arrays.

    Returns:
        list: List of annotated frame images with indices overlayed.
    """
    annotated_frames = []
    for i, frame in enumerate(frames):
        # Create a copy of the frame for annotation.
        annotated_frame = frame.copy()
        cv2.putText(
            annotated_frame,
            f"{i}",  # The index
            (10, 20),  # Position (top-left corner)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.6,  # Font scale
            (1.0, 1.0, 1.0),  # Text color (white)
            2,  # Thickness
            cv2.LINE_AA,  # Anti-aliased
        )
        annotated_frames.append(annotated_frame)
    return annotated_frames

def save_video(video: torch.Tensor, path: str):
    """
    Save a video tensor to a video file.

    This function processes the video tensor by interpolating to ensure even dimensions,
    rearranges the tensor, creates frame grids, and writes the video to the specified path.

    Args:
        video (torch.Tensor): Video tensor of shape (n, c, t, h, w).
        path (str): Destination file path to save the video.

    Returns:
        str: The path where the video was saved.
    """
    n, c, t, h, w = video.shape
    video = torch.nn.functional.interpolate(
        rearrange(video, "n c t h w -> (n t) c h w"), (h // 2 * 2, w // 2 * 2), mode="bilinear"
    )
    video = rearrange(video, "(n t) c h w -> t n c h w", n=n, t=t)
    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(1), padding=0) for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.io.write_video(path, grid, fps=7, video_codec="h264", options={"crf": "10"})

    return path

def multicond_comparison_app():
    """
    Create and return the Gradio interface for multi-condition video comparison.

    This function sets up the UI components for the demo application, including video displays,
    input controls, model selection, and buttons for generating videos. It also registers the event
    handlers for the interface.

    Returns:
        gr.Blocks: The Gradio Blocks interface for the application.
    """
    model_names, model_metadata = load_models()
    if args.detect_models:
        detected_model_names, detected_models = detect_models()
        model_names.extend(detected_model_names)
        model_metadata.update(detected_models)

    image2video = Image2Video(args.result_dir, args.model_meta_path, args.camera_pose_meta_path, device=args.device, model_meta_data=model_metadata)
    dataset = load_dataset()

    with gr.Blocks(analytics_enabled=False) as app_interface:
      
        with gr.Row():
            gt_video = gr.Video(label="Ground Truth Video", elem_id="gt_vid", interactive=False, autoplay=True, loop=True)
            cam_traj_vis = gr.Model3D(label="Camera Trajectory", elem_id="cam_traj", clear_color=[1.0, 1.0, 1.0, 1.0])
        with gr.Row():
            frame_gallery = gr.Gallery(label="Frames (Indexed)", columns=5)
        with gr.Row():
            reference_selector = gr.CheckboxGroup(
                choices=[], 
                label="Select Reference frame", 
                info="Select single frame to start generation from"
            )
            frame_selector = gr.CheckboxGroup(
                choices=[], 
                label="Select Condition Frames", 
                info="Select frames by index"
            )

        with gr.Row():
            video_dropdown = gr.Dropdown(label="Video Selection", elem_id="video_dropdown", choices=dataset.get_all_sample_names())

        with gr.Row():
            with gr.Column():
                model_dropdown1 = gr.Dropdown(label="Model 1", elem_id='model_dd1', choices=model_names)
                gen_vid1 = gr.Video(label="Generated Video 1", elem_id="gen_vid1", interactive=False, autoplay=True, loop=True)
                gen_btn1 = gr.Button("Generate")
            with gr.Column():
                model_dropdown2 = gr.Dropdown(label="Model 2", elem_id='model_dd1', choices=model_names)
                gen_vid2 = gr.Video(label="Generated Video 2", elem_id="gen_vid2", interactive=False, autoplay=True, loop=True)
                gen_btn2 = gr.Button("Generate")
        with gr.Row(equal_height=True):
            input_text = gr.Textbox(label='Prompts', scale=4)
        with gr.Row():
            negative_prompt = gr.Textbox(label='Negative Prompts', value="Fast movement, jittery motion, abrupt transitions, distorted body, missing limbs, unnatural posture, blurry, cropped, extra limbs, bad anatomy, deformed, glitchy motion, artifacts.")

        with gr.Row():
            with gr.Column():
                camera_pose_type = gr.Dropdown(label='Camera Pose Type', elem_id="camera_pose_type", choices=load_camera_pose_type())
                trace_extract_ratio = gr.Slider(minimum=0, maximum=1.0, step=0.1, elem_id="trace_extract_ratio", label="Trace Extract Ratio", value=0.1)
                trace_scale_factor = gr.Slider(minimum=0, maximum=5, step=0.1, elem_id="trace_scale_factor", label="Camera Trace Scale Factor", value=1.0)
                auto_reg_steps = gr.Slider(minimum=0, maximum=10, step=1, elem_id="auto_reg_steps", label="Auto-regressive Steps", value=0)
            with gr.Column():
                enable_camera_condition = gr.Checkbox(label='Enable Camera Condition', elem_id="enable_camera_condition", value=True)
                camera_cfg = gr.Slider(minimum=1.0, maximum=4.0, step=0.1, elem_id="Camera CFG", label="Camera CFG", value=1.0, visible=False)
                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=3.5, elem_id="cfg_scale")
                frame_stride = gr.Slider(minimum=1, maximum=10, step=1, label='Frame Stride', value=2, elem_id="frame_stride")
                steps = gr.Slider(minimum=1, maximum=250, step=1, elem_id="steps", label="Sampling Steps (DDPM)", value=25)
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**31, step=1, value=12333)

        def generate(reference_index, cond_frame_indices, video_name, *inputs):
            """
            Generate a video based on the selected reference and condition frames along with user inputs.

            Args:
                reference_index (list): List containing the reference frame index.
                cond_frame_indices (list): List of condition frame indices.
                video_name (str): Selected video name from the dataset.
                *inputs: Additional inputs including model selection, prompts, camera settings, etc.

            Returns:
                tuple: Generated video and camera trajectory visualization.
            """
            index = dataset.get_index_by_name(video_name)
            batch = dataset[index]
            if len(cond_frame_indices) > 0:
                cond_frame_indices = [int(ind) for ind in cond_frame_indices]
                add_cond_frames = batch['video'][:, cond_frame_indices]
                batch["cond_frames"] = add_cond_frames
                batch["RT_cond"] = batch["RT"][cond_frame_indices]
            ref_index = int(reference_index[0]) if len(reference_index) > 0 else 0
            batch['video'] = batch['video'][:, ref_index:]
            batch['RT'] = batch['RT'][ref_index:]
            inputs = list(inputs[:1]) + [None] + list(inputs[1:])
            return image2video.get_image(*inputs, batch=batch)
        
        def load_video(video_name):
            """
            Load the ground truth video and prepare frame indices for selection.

            Args:
                video_name (str): Name of the video to load.

            Returns:
                tuple: Path to the saved ground truth video, list of annotated frames, 
                       updated choices for frame selector, and updated choices for reference selector.
            """
            index = dataset.get_index_by_name(video_name)
            batch = dataset[index]
            video = batch['video']
            video_path = './current_gt.mp4'
            save_video(video.unsqueeze(0), video_path)
            video = video.permute(1, 2, 3, 0).numpy()
            video = np.clip((video + 1) / 2, 0, 1)
            indices = [str(i) for i in range(video.shape[0])]
            images = overlay_indices([v for v in video]) if not args.no_overlay_indices else [v for v in video]
            return video_path, images, gr.update(choices=indices), gr.update(choices=indices)
        
        # Set up event for video dropdown selection.
        video_dropdown.select(fn=load_video, inputs=[video_dropdown], outputs=[gt_video, frame_gallery, frame_selector, reference_selector])

        # Set up events for the generate buttons.
        gen_btn1.click(
            fn=generate,
            inputs=[reference_selector, frame_selector, video_dropdown, model_dropdown1, input_text, negative_prompt, camera_pose_type, trace_extract_ratio, frame_stride, steps, trace_scale_factor, camera_cfg, cfg_scale, seed, enable_camera_condition, auto_reg_steps],
            outputs=[gen_vid1, cam_traj_vis],
        )

        gen_btn2.click(
            fn=generate,
            inputs=[reference_selector, frame_selector, video_dropdown, model_dropdown2, input_text, negative_prompt, camera_pose_type, trace_extract_ratio, frame_stride, steps, trace_scale_factor, camera_cfg, cfg_scale, seed, enable_camera_condition, auto_reg_steps],
            outputs=[gen_vid2, cam_traj_vis],
        )

    return app_interface

def get_ip_addr():
    """
    Retrieve the host's IP address for Gradio server binding.

    Returns:
        str or None: The IP address if successfully retrieved, otherwise None.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]
    except:
        return None

if __name__ == "__main__":
    app = multicond_comparison_app()
    app.queue(max_size=12)
    app.launch(max_threads=10, server_name=get_ip_addr() if args.use_host_ip else None, allowed_paths=["gradio", "internal"])
