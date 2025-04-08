import os
import random
from typing import Literal, Union, List, Tuple
import json
import logging
import numpy as np
import omegaconf
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

mainlogger = logging.getLogger('mainlogger')


class RealEstate10K(Dataset):
    """
    RealEstate10K Dataset.
    For each video, its meta info is stored in a txt file whose contents are as follows:
    line 0: video_url
    line 1: empty
    line 2: caption

    In the rest, each line is a frame, including frame path, 4 camera intrinsics, and 3*4 camera pose (the matrix is row-major order).

    e.g.
    line 3: 0_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    line 4: 1_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    ...

    meta_path: path to the meta file
    meat_list: path to the meta list file
    data_dir: path to the data folder
    video_length: length of the video clip for training
    resolution: target resolution, a list of int, [h, w]
    frame_stride: stride between frames, int or list of int, [min, max], do not larger than 32 when video_length=16
    spatial_transform: spatial transformation, ["random_crop", "resize_center_crop"]
    count_globalsteps: whether to count global steps
    bs_per_gpu: batch size per gpu, used to count global steps

    """

    def __init__(self,
                 meta_path,
                 meta_list,
                 data_dir,
                 caption_file,
                 per_frame_scale_path=None,
                 camera_pose_sections=1,
                 video_length=16,
                 resolution=[256, 256],  # H, W
                 frame_stride=1,  # [min, max], do not larger than 32 when video_length=16
                 frame_stride_for_condition=0,
                 invert_video=False,
                 spatial_transform=None,
                 count_globalsteps=False,
                 bs_per_gpu=None,
                 RT_norm=False,
                 load_raw_resolution=True,
                 return_full_clip=False,
                 additional_cond_frames: Literal['none', 'random', 'last', "random_true", "random_offset", "random_full"]='none',
                 num_additional_cond_frames: Union[List[int], int]=0,
                 exclude_samples: List[str] = [],
                 adaptive_sampling_range: Tuple[int,int] = None
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride_for_condition = frame_stride_for_condition
        self.frame_stride = frame_stride
        self.spatial_transform_type = spatial_transform
        self.additional_cond_frames = additional_cond_frames
        self.return_full_clip = return_full_clip
        self.adaptive_sampling_range = adaptive_sampling_range
        self.num_additional_cond_frames = list(num_additional_cond_frames) if not isinstance(num_additional_cond_frames, int) else num_additional_cond_frames
        assert self.spatial_transform_type in ['resize_center_crop']

        self.count_globalsteps = count_globalsteps
        self.bs_per_gpu = bs_per_gpu
        self.invert_video = invert_video
        self.RT_norm = RT_norm
        self.load_raw_resolution = load_raw_resolution
        self.camera_pose_sections = camera_pose_sections

        self.metadata = []
        with open(meta_list, 'r') as f:
            # self.metadata = [line.strip() for line in f.readlines()]
            self.metadata = np.array([line.strip() for line in f.readlines()], dtype=np.string_)

        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        mainlogger.info(f"Found {len(self.captions)} captions")

        if per_frame_scale_path:
            self.per_frame_scale = np.load(per_frame_scale_path, allow_pickle=True)['arr_0'].item()
        mainlogger.info(f'Dataset initialized!')
        mainlogger.info(f'Data directory: {data_dir}')
        mainlogger.info(f'Dataset length: {len(self.metadata)}')

        self.invalid_samples = set(exclude_samples)
        
        
        valid_metadata = []
        if len(self.invalid_samples) > 0:
            for i in range(self.metadata.shape[0]):
                if self.metadata[i].decode("utf-8") not in self.invalid_samples:
                    valid_metadata.append(self.metadata[i])
            valid_metadata = np.stack(valid_metadata, dtype=np.string_)
            self.metadata = valid_metadata

    def _resize_for_rectangle_crop(self, frames, H, W, fx, fy, cx, cy):
        '''
        :param frames: C,F,H,W
        :param image_size: H,W
        :return: frames: C,F,crop_H,crop_W;  camera_intrinsics: F,3,3
        '''
        ori_H, ori_W = frames.shape[-2:]
        if ori_W / ori_H > W / H:
            frames = transforms.functional.resize(
                frames,
                size=[H, int(ori_W * H / ori_H)],
            )
        else:
            frames = transforms.functional.resize(
                frames,
                size=[int(ori_H * W / ori_W), W],
            )

        resized_H, resized_W = frames.shape[2], frames.shape[3]
        frames = frames.squeeze(0)

        delta_H = resized_H - H
        delta_W = resized_W - W

        top, left = delta_H // 2, delta_W // 2
        frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

        fx = fx * resized_W
        fy = fy * resized_H
        cx = cx * W
        cy = cy * H
        _1, _0 = torch.ones_like(fx), torch.zeros_like(fx)
        camera_intrinsics = torch.hstack([fx, _0, cx, _0, fy, cy, _0, _0, _1]).reshape(-1, 3, 3)  # [F, 3, 3]

        return frames, camera_intrinsics, resized_H, resized_W

    def __getitem__(self, index):
        
        to_inverse = (self.invert_video and random.random() > 0.5)

        ## get frames until success
        index = index % len(self.metadata)
        sample_name = self.metadata[index].decode('utf-8')
        if sample_name in self.invalid_samples:
            return self.__getitem__(random.randint(0, len(self)))
        with open(f"{self.meta_path}/{sample_name}.txt", 'r') as f:
            lines = f.readlines()
        
        cap_name = f"{sample_name}.mp4"
        if cap_name not in self.captions:
            self.invalid_samples.add(sample_name)
            mainlogger.warning(f"Invalid sample {sample_name} (Total: {len(self.invalid_samples)}): No caption found.")
            return self.__getitem__(random.randint(0, len(self)))
        else:
            caption = self.captions[cap_name][0]
        video_path = os.path.join(self.data_dir, f'{sample_name}.mp4')
        if not os.path.exists(video_path):
            return self.__getitem__(random.randint(0, len(self)))
        
        try:
            if self.load_raw_resolution:
                video_reader = VideoReader(video_path, ctx=cpu(0))
            else:
                video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                #assert len(video_reader) < self.video_length, print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
        except Exception as e:
            print(f"Error occurred when initializing video reader for {video_path}:\n{e}")
            return self.__getitem__(random.randint(0, len(self)))

        fps_ori = video_reader.get_avg_fps()
        lines = lines[1:]
        frame_num = len(lines)

        frame_stride_drop = 0
        while True:
            if isinstance(self.frame_stride, int):
                frame_stride = max(self.frame_stride - frame_stride_drop, 1)
            elif (isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig)) and len(self.frame_stride) == 2:  # [min, max]
                assert (self.frame_stride[0] <= self.frame_stride[1]), f"frame_stride[0]({self.frame_stride[0]}) > frame_stride[1]({self.frame_stride[1]})"
                frame_stride = random.randint(self.frame_stride[0], self.frame_stride[1])
            else:
                print(type(self.frame_stride))
                print(len(self.frame_stride))
                print(f"frame_stride={self.frame_stride}")
                raise NotImplementedError

            required_frame_num = frame_stride * (self.video_length - 1) + 1
            if frame_num < required_frame_num:
                if isinstance(self.frame_stride, int) and frame_num < required_frame_num * 0.5:
                    frame_stride_drop += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            break
          
        ## select a random clip
        if self.video_length != -1:
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
        else:
            frame_indices = list(range(frame_num))
            frame_indices = frame_indices[::frame_stride]

        camera_data = torch.from_numpy(np.loadtxt(lines))[frame_indices].float()  # [t, ]
        fx, fy, cx, cy = camera_data[:, 1:5].chunk(4, dim=-1)  # [t,4]
        camera_pose_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
        camera_pose_4x4 = torch.cat([camera_pose_3x4, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * len(frame_indices))], dim=1)  # [t, 4, 4]

        camera_pose_4x4_cond = torch.zeros(1)
        try:
            frames = video_reader.get_batch(frame_indices)

            if self.additional_cond_frames is not None and self.additional_cond_frames!='none':
                context_indices = self.sample_context_indices(
                    strategy = self.additional_cond_frames,
                    clip_range = (frame_indices[0], frame_indices[-1]),
                    video_length = len(video_reader),
                    stride = frame_stride
                )
                add_cond_frames = video_reader.get_batch(context_indices)
                frames = np.concatenate([frames.asnumpy(), add_cond_frames.asnumpy()], axis=0)
                camera_data_cond = torch.from_numpy(np.loadtxt(lines))[context_indices].float()  # [t, ]
                camera_pose_3x4_cond = camera_data_cond[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
                camera_pose_4x4_cond = torch.cat([camera_pose_3x4_cond, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * len(context_indices))], dim=1)  # [t, 4, 4]
            
            else:
                frames = frames.asnumpy()
        except Exception as e:
            del video_reader
            self.invalid_samples.add(sample_name)
            mainlogger.warning(f"Invalid sample {sample_name}(Total: {len(self.invalid_samples)}): {e}")
            #print(f"Error when reading frames for video {sample_name}:\n{e}")
            return self.__getitem__(random.randint(0, len(self)))
        
        all_frames = []
        if self.return_full_clip:
            all_frames = video_reader[:].asnumpy()

        del video_reader

        ## process data
        #assert (frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

        ## spatial transformations
        if self.spatial_transform_type == 'resize_center_crop':
            frames, camera_intrinsics, resized_H, resized_W = self._resize_for_rectangle_crop(
                frames,
                self.resolution[0], self.resolution[1],  # H, W
                fx, fy, cx, cy
            )
            camera_data[:, 1:5] = torch.stack(
                [
                    camera_intrinsics[:, 0, 0],  # fx
                    camera_intrinsics[:, 1, 1],  # fy
                    camera_intrinsics[:, 0, 2],  # cx
                    camera_intrinsics[:, 1, 2],  # cy
                ],
                dim=-1,
            )

        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'

        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // max(1, frame_stride)

        add_cond_frames = torch.zeros(1)
        if self.additional_cond_frames is not None and self.additional_cond_frames != 'none':
            add_cond_frames = frames[:, -len(context_indices):]
            add_cond_frames = add_cond_frames.permute(1,0,2,3) # t, c, h, w 
            frames = frames[:, :-len(context_indices)]
        else:
            add_cond_frames = torch.zeros((1))

        if to_inverse:
            # inverse frame order in dim=1
            frames = frames.flip(dims=(1,))

        data = {
            'video': frames,  # [c,t,h,w]
            'caption': caption,
            'video_path': video_path,
            'fps': fps_clip,
            'frame_stride': frame_stride if self.frame_stride_for_condition == 0 else self.frame_stride_for_condition,
            'RT': camera_pose_4x4,  # Tx4x4
            'camera_data': camera_data,
            'camera_intrinsics': camera_intrinsics,  # Tx3x3
            'cond_frames': add_cond_frames,
            'RT_cond': camera_pose_4x4_cond,
            'all_frames': all_frames,
            # 'trajs': torch.zeros(2, self.video_length, frames.shape[2], frames.shape[3])
        }

        if hasattr(self, "per_frame_scale"):
            data['per_frame_scale'] = torch.from_numpy(self.per_frame_scale[sample_name][frame_indices]).float()
        return data
    
    def sample_context_indices(self, 
                                strategy: str,
                                stride: int,
                                clip_range: Tuple[int,int], 
                                video_length: int,
                                offset: int = 0,
                                stride_corrected: bool = True):
        if isinstance(self.num_additional_cond_frames, list) or isinstance(self.num_additional_cond_frames, tuple):
            num_add_cond_frames = self.num_additional_cond_frames[-1]
        else:
            num_add_cond_frames = self.num_additional_cond_frames

        if strategy == "random_full":
            potential_indices = np.arange(video_length)
        elif strategy == "random_outside":
            potential_indices_front = np.arange(clip_range[0]-offset)
            potential_indices_back = np.arange(clip_range[1]+offset, video_length)
            potential_indices = np.concatenate([potential_indices_front, potential_indices_back])
        elif strategy == "random_back":
            potential_indices = np.arange(clip_range[-1]+offset, video_length)
        elif strategy == "random_front":
            potential_indices = np.arange(0, clip_range[0]-offset)
        elif strategy == "last":
            stride_corrected = False
            potential_indices = np.array([clip_range[-1]+offset])
        elif strategy == "furthest_distance":
            stride_corrected = False
            dist_front = clip_range[0]
            dist_back = video_length - clip_range[-1]
            potential_indices = np.zeros(1) if dist_front > dist_back else np.ones(1)*(video_length-1)
        else:
            mainlogger.error(f"Unknown strategy '{strategy}' to sample context frames.")
        if stride_corrected == True and stride >= 0:
            potential_indices = potential_indices[::stride]
        
        num_add_cond_frames = min(potential_indices.shape[0], num_add_cond_frames)
        np.random.shuffle(potential_indices)
        context_indices = potential_indices[:num_add_cond_frames]
        return context_indices
        


    def custom_collate_fn(self, batch):
        """
        Custom collate function to subsample a consistent number of `cond_frames` for the entire batch,
        while using the default collate function for all other keys.
        """
        
        if self.additional_cond_frames != 'none' and isinstance(self.num_additional_cond_frames, list):
            num_cond_frames = random.randint(self.num_additional_cond_frames[0], self.num_additional_cond_frames[1])
            min_num_cond_frames = np.min([s['cond_frames'].shape[0] for s in batch])
            num_cond_frames = min(num_cond_frames, min_num_cond_frames)
            for sample in batch:
                sample['cond_frames'] = sample['cond_frames'][:num_cond_frames]
                sample['RT_cond'] = sample['RT_cond'][:num_cond_frames]

        return default_collate(batch)
    
    def get_all_sample_names(self):
        return [self.metadata[i].decode('utf-8') for i in range(len(self.metadata))]
    
    def get_index_by_name(self, name):
        for i in range(len(self.metadata)):
            if self.metadata[i].decode('utf-8') == name:
                return i

    def __len__(self):
        return len(self.metadata)
