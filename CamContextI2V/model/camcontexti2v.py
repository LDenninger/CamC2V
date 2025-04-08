import logging
from math import sqrt

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from typing import Literal
from functools import partial
import re
import time
import copy

from model.base import CameraControlLVDM
from model.modules.modified_forwards import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
    new_forward_for_unet,
    new_efficient_forward_for_cross_attention,
    new_forward_for_cross_attention
)
from model.modules.utils import CrossNormalization
from model.modules.epipolar import Epipolar, pix2coord
from utils.utils import instantiate_from_config, human_readable_number
from lvdm.common import extract_into_tensor, default

mainlogger = logging.getLogger('mainlogger')

class CamContextI2V(CameraControlLVDM):
    """
    
    Parameters:
        add_type (str): Type of to add the plucker embeddings into the UNet.
        multi_cond_strategy (str): Strategy for combining multiple context condition images.
    """
    def __init__(self,
                    add_type="add_into_temporal_attn",
                    multi_cond_strategy: Literal['max', 'avg', 'token_concat'] = None,
                    epipolar_config=None,
                    pose_guided_cond_encoder_config=None,
                    plucker_proj_trainable: bool = False,
                    epipolar_attn_trainable: bool = False,
                    pose_guided_cond_trainable: bool = False,
                    multi_cond_adaptor_trainable: bool = False,
                    first_unet_block_trainable: bool = False,
                    first_unet_block_freeze_steps: int = 0,
                    use_cross_normalization: bool = False,
                    use_zero_conv_latent_input: bool = False,
                    use_semantic_branch: bool = True,
                    epipolar_mask_freeze_steps: int = None,
                    cross_normalization_mode: Literal["token", "spatio_temporal"] = "spatio_temporal",
                    use_pose_embedding_in_latent_adaptor: bool = False,
                    inject_trainable_lora_unet : bool = False,
                    lora_config: dict = {},
                    *args, **kwargs):

        super(CamContextI2V, self).__init__(*args, **kwargs)
        self.add_type = add_type
        self.pose_agent_enc = None # initialized in _get_multi_cond_func()

        ##-- Training Parameters --##
        self.plucker_proj_trainable = plucker_proj_trainable
        self.epipolar_attn_trainable = epipolar_attn_trainable
        self.pose_guided_cond_trainable = pose_guided_cond_trainable
        self.multi_cond_adaptor_trainable = multi_cond_adaptor_trainable
        self.first_unet_block_trainable = first_unet_block_trainable
        self.first_unet_block_freeze_steps = first_unet_block_freeze_steps
        self.epipolar_mask_freeze_steps = epipolar_mask_freeze_steps
        self.use_pose_embedding_in_latent_adaptor = use_pose_embedding_in_latent_adaptor
        self.use_zero_conv_latent_input = use_zero_conv_latent_input
        self.cross_normalization_mode = cross_normalization_mode
        self.use_semantic_branch = use_semantic_branch

        # Set up additional context conditioning
        self.multi_cond_strategy = multi_cond_strategy
        if self.multi_cond_strategy is not None:
            self.multi_cond_func = self._get_multi_cond_func(pose_guided_cond_encoder_config, pose_guided_cond_trainable=pose_guided_cond_trainable, *args, **kwargs)
            self.use_cross_normalization = use_cross_normalization
            self.cond_cross_norm = CrossNormalization((-3, -2, -1))
        if self.use_zero_conv_latent_input:
            self.multi_cond_in_projection = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
            nn.init.constant_(self.multi_cond_in_projection.weight, 0.0)
            nn.init.constant_(self.multi_cond_in_projection.bias, 0.0)
            
            self.multi_cond_in_projection.requires_grad_(True)

        self.epipolar_config = epipolar_config
        if self.epipolar_config is not None:
            if not hasattr(self.epipolar_config, "is_3d_full_attn"):
                self.epipolar_config.is_3d_full_attn = False
            if not hasattr(self.epipolar_config, "attention_resolution"):
                self.epipolar_config.attention_resolution = [8, 4, 2, 1]
            if not hasattr(self.epipolar_config, "apply_epipolar_soft_mask"):
                self.epipolar_config.apply_epipolar_soft_mask = False
            if not hasattr(self.epipolar_config, "soft_mask_temperature"):
                self.epipolar_config.soft_mask_temperature = 1.0
            if not hasattr(self.epipolar_config, "epipolar_hybrid_attention"):
                self.epipolar_config.epipolar_hybrid_attention = False
            if not hasattr(self.epipolar_config, "epipolar_hybrid_attention_v2"):
                self.epipolar_config.epipolar_hybrid_attention_v2 = False
            if not hasattr(self.epipolar_config, "only_self_pixel_on_current_frame"):
                self.epipolar_config.only_self_pixel_on_current_frame = False
            if not hasattr(self.epipolar_config, "current_frame_as_register_token"):
                self.epipolar_config.current_frame_as_register_token = False
            if not hasattr(self.epipolar_config, "pluker_add_type"):
                self.epipolar_config.pluker_add_type = "add_to_pre_x_only"
            if not hasattr(self.epipolar_config, "add_small_perturbation_on_zero_T"):
                self.epipolar_config.add_small_perturbation_on_zero_T = False

        bound_method = new_forward_for_unet.__get__(
            self.model.diffusion_model,
            self.model.diffusion_model.__class__
        )
        setattr(self.model.diffusion_model, 'forward', bound_method)

        spatial_cross_attn_pattern = r'\.\d\.1\..*\.attn2'
        running_block_index = 0
        if "diffusion_model_trainable_param_list" in kwargs:
            diffusion_model_trainable_param_list = kwargs.get("diffusion_model_trainable_param_list")
            for name, param in self.model.diffusion_model.named_parameters():
                if name in diffusion_model_trainable_param_list:
                    param.requires_grad_(True)

        for _name, _module in self.model.diffusion_model.named_modules():
            if _module.__class__.__name__ == 'TemporalTransformer':
                bound_method = new_forward_for_TemporalTransformer.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == 'TimestepEmbedSequential':
                bound_method = new_forward_for_TimestepEmbedSequential.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == "CrossAttention" and self.pose_agent_enc is not None and re.search(spatial_cross_attn_pattern, _name):
                mainlogger.info(f"Adjusting forward of {_name} to information retrieval from pose-guided encoder")
                cond_encoder_forward_bound_method = partial(self.pose_agent_enc.forward, block_index=running_block_index)
                bound_method_eff = new_efficient_forward_for_cross_attention.__get__(_module, _module.__class__)
                setattr(_module, 'forward', partial(bound_method_eff, image_emb_func=cond_encoder_forward_bound_method))
                bound_method = new_forward_for_cross_attention.__get__(_module, _module.__class__)
                setattr(_module, 'efficient_forward', partial(bound_method, image_emb_func=cond_encoder_forward_bound_method))
                running_block_index += 1

            elif _module.__class__.__name__ == 'BasicTransformerBlock':
                # SpatialTransformer only
                if _module.context_dim is None and _module.attn1.to_k.in_features != self.model.diffusion_model.init_attn[0].proj_in.out_channels:  # BasicTransformerBlock of TemporalTransformer, only self attn, context_dim=None

                    bound_method = new_forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, 'forward', bound_method)

                    bound_method = new__forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

                    if self.pose_encoder is not None:
                        pluker_projection = nn.Linear(_module.attn1.to_k.in_features, _module.attn1.to_k.in_features)
                        nn.init.zeros_(list(pluker_projection.parameters())[0])
                        nn.init.zeros_(list(pluker_projection.parameters())[1])
                        pluker_projection.requires_grad_(plucker_proj_trainable )
                        _module.add_module('pluker_projection', pluker_projection)
                        # _module.add_module('norm_pluker1', nn.LayerNorm(_module.attn1.to_k.in_features))
                        # _module.add_module('norm_pluker2', nn.LayerNorm(_module.attn1.to_k.in_features))

                    if self.epipolar_config is not None:
                        epipolar = Epipolar(
                            query_dim=_module.attn1.to_k.in_features,
                            context_dim=_module.attn1.to_k.in_features,
                            heads=_module.attn1.heads,
                            **self.epipolar_config
                        )
                        _module.add_module('epipolar', epipolar)
                        for _, param in epipolar.named_parameters():
                            param.requires_grad_(epipolar_attn_trainable)
                        # _module.add_module('norm_epipolar', nn.LayerNorm(_module.attn1.to_k.in_features))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_relative_c2w_RT_pairs(self, RT: Tensor):
        '''
        :param RT: B, T, 4 4   c2w relative RT
        :return: relative RT pairs, c2w, (B, T, T, 4, 4)
        given c2w RT, camera system transform from T1 to T2: inverse(RT_2) @ (RT_1)
        '''

        RT_inv = rearrange(RT.inverse(), "b t ... -> b 1 t ...")
        relative_RT_pairs = RT_inv @ rearrange(RT, "b t ... -> b t 1 ...")  # B, T, T, 4, 4

        return relative_RT_pairs  # B,T,T,4,4

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_fundamental_matrix(self, K: Tensor, R: Tensor, t: Tensor) -> Tensor:
        '''
        :param   K: B, 3, 3
        :param   R: B, 3, 3
        :param   t: B, 3, 1
        :return: F: B, 3, 3
        '''
        E = torch.cross(t, R, dim=-2)
        K_inv = torch.inverse(K)
        F = K_inv.transpose(-1, -2) @ E @ K_inv
        return F

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_epipolar_mask(self, F: Tensor, T: int, H: int, W: int, downsample: int, final_rearrange: bool = True):
        """
        modified to take in batch inputs

        Args:
            grid: (H*W, 3)
            F: camera fundamental matrix (B, T1, T2, 3, 3)
            resolution: feature map resolution H * W
            downsample: downsample scale

        return: weight matrix M(HW * HW)
        """
        # B = F.shape[0]
        device = F.device

        y = torch.arange(0, H, dtype=torch.float, device=device)  # 0 .. 128
        x = torch.arange(0, W, dtype=torch.float, device=device)  # 0 .. 84

        y = pix2coord(y, downsample)  # H
        x = pix2coord(x, downsample)  # W

        grid_y, grid_x = torch.meshgrid(y, x)  # H * W
        # grid_y: 84x128
        # 3 x HW·
        # TODO check whether yx or xy
        grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=2).view(-1, 3).float()  # H*W, 3

        lines = F @ grid.transpose(-1, -2)  # [B, T1, T2, 3, H*W]
        norm = torch.norm(lines[..., :2, :], dim=-2, keepdim=True)  # [B, T1, T2, 1, H*W]
        # norm = torch.where(
        #     norm == 0.0,
        #     torch.ones_like(norm),
        #     norm
        # )
        lines = lines / norm  # [B, T1, T2, 3, H*W]

        dist = (lines.transpose(-1, -2) @ grid.transpose(-1, -2)).abs()  # [B, T1, T2, H*W, H*W]
        mask = dist < (downsample * sqrt(2) / 2)  # [B, T1, T2, H*W, H*W]
        # switch to 3d full attention if epipolar mask is empty
        if self.epipolar_config.apply_epipolar_soft_mask:
            raise NotImplementedError
            mask = -dist * self.epipolar_config.soft_mask_temperature  # 高斯分布形式的权重

        if self.epipolar_config.epipolar_hybrid_attention:    # Handling Empty Epipolar Masks
            mask = torch.where(mask.any(dim=-1, keepdim=True), mask, torch.ones_like(mask))

        if self.epipolar_config.epipolar_hybrid_attention_v2:  # Handling Empty Epipolar Masks
            mask = torch.where(mask.any(dim=[2,4], keepdim=True).repeat(1,1,T,1,H*W), mask, torch.ones_like(mask))

        if self.epipolar_config.only_self_pixel_on_current_frame:
            # Step 1: Zero out masks for same frame interactions
            same_frame = torch.eye(T, device=device, dtype=mask.dtype).view(1, T, T, 1, 1)
            mask = mask * (~same_frame)  # Zero out same frame interactions

            # Step 2: Create identity mask for same pixel in the same frame
            identity_hw = torch.eye(T * H * W, device=device, dtype=mask.dtype).reshape(T, H, W, T, H, W)
            identity_hw = rearrange(
                identity_hw,
                'T1 H1 W1 T2 H2 W2 -> 1 T1 T2 (H1 W1) (H2 W2)'
            ).repeat(mask.shape[0], 1, 1, 1, 1)
            mask = torch.where(identity_hw, identity_hw, mask)

        if self.epipolar_config.current_frame_as_register_token:
            # Step 1: Zero out masks for same frame interactions
            same_frame = torch.eye(T, device=device, dtype=mask.dtype).view(1, T, T, 1, 1).repeat(mask.shape[0], 1, 1, H * W, H * W)
            mask = torch.where(same_frame, same_frame, mask)
        if final_rearrange:
            return rearrange(mask, "B T1 T2 HW1 HW2 -> B (T1 HW1) (T2 HW2)")
        else:
            return mask

    def add_small_perturbation(self, t, epsilon=1e-6):
        zero_mask = (t.abs() < epsilon).all(dim=-2, keepdim=True)  # 检查 T 的 x, y, z 是否都接近 0
        perturbation = torch.randn_like(t) * epsilon  # 生成微小扰动
        t = torch.where(zero_mask, perturbation, t)  # 如果 T 为零，替换为扰动，否则保持原值

        return t
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False,
                    return_cond_frame_index=False, return_cond_frame=False, return_original_input=False, rand_cond_frame=None,
                    enable_camera_condition=True, return_camera_data=False, return_video_path=False, return_depth_scale=False,
                    trace_scale_factor=1.0, cond_frame_index=None, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        T = x.shape[2]
 
        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]
        

        ## get conditioning frame
        if cond_frame_index is None:
            cond_frame_index = torch.zeros(batch_size, device=device, dtype=torch.long)
            rand_cond_frame = self.rand_cond_frame if rand_cond_frame is None else rand_cond_frame
            if rand_cond_frame: 
                cond_frame_index = torch.randint(0, self.model.diffusion_model.temporal_length, (batch_size,), device=device)


        ###################################################################################################################################
        ########################################### only change here, add camera_condition input ##########################################
        depth_scale = torch.ones((batch_size,), device=device)
        if enable_camera_condition:
            camera_condition_log, camera_condition_kwargs = self.get_batch_input_camera_condition_process(
                batch, x, cond_frame_index, trace_scale_factor, rand_cond_frame
            )
            if "depth_scale" in camera_condition_log:
                depth_scale = camera_condition_log["depth_scale"]
        ########################################### only change here, add camera_condition input ###########################################
        ###################################################################################################################################
        ###################################################################################################################################
        ########################################### only change here, add multi condition input ###########################################

        cond_frames = None
        if 'cond_frames' in batch and batch['cond_frames'] is not None:
            cond_frames = super().get_input(batch, 'cond_frames')
            if self.multi_cond_strategy in ['token_concat_latent', 'token_concat_latent_epipolar']:
                x = torch.cat([x, rearrange(cond_frames, 'B C D H W -> B D C H W')], dim=2)

        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
        
        if self.multi_cond_strategy == 'token_concat_latent':
            
            z_cond = z[torch.arange(cond_frame_index.shape[0]),:,cond_frame_index].unsqueeze(2)
            z_add = z[:,:,-cond_frames.shape[2]:]
            z_inp = torch.cat([z_cond, z_add], dim=2)
            z_inp = rearrange(z_inp, 'B C D H W -> B (C H W) D') # TODO: this might be wrong see below for the correct way
            img_cat_cond = self.multi_cond_latent_adaptor(z_inp)
            img_cat_cond = rearrange(img_cat_cond, 'B (H W) D -> B D H W', H=32, W=32)
            x = x[:,:,:-cond_frames.shape[1]]
            z = z[:,:,:-cond_frames.shape[1]]
        elif self.multi_cond_strategy == 'token_concat_latent_epipolar':
            z_cond = z[torch.arange(cond_frame_index.shape[0]),:,cond_frame_index].unsqueeze(2)
            z_add = z[:,:,-cond_frames.shape[1]:]
            z_inp = torch.cat([z_cond, z_add], dim=2)
            z_inp = rearrange(z_inp, 'B D C H W -> B (C H W) D')

            x = x[:,:,:-cond_frames.shape[1]]
            z = z[:,:,:-cond_frames.shape[1]]
            _, H, W = x.shape[2:]


            H_enc, W_enc = z_add.shape[-2:]
            epipolar_mask = None
            if self.multi_cond_latent_adaptor.use_mask: 
                epipolar_mask = self.compute_conditional_epipolar_mask(batch, H, W, return_squeezed_tokens=True, cond_frame_indices=cond_frame_index)

            plucker_embedding = None
            if self.multi_cond_latent_adaptor.use_plucker_embedding:
                plucker_embedding = camera_condition_kwargs['camera_condition']['pluker_embedding_features'][0]
                plucker_embedding = rearrange(plucker_embedding, "B D T H W -> B (T H W) D")

            img_cat_cond = self.multi_cond_latent_adaptor(z_inp, epipolar_mask, plucker_embedding_features=plucker_embedding)
            if self.use_cross_normalization or self.use_zero_conv_latent_input:
                original_cond = z[torch.arange(batch_size, device=device), :, cond_frame_index, :, :]
                if self.use_cross_normalization:
                    if self.cross_normalization_mode == "spatio_temporal":
                        img_cat_cond = rearrange(img_cat_cond, 'B (T H W) D -> B T D H W', T=T, H=H_enc, W=W_enc)
                        img_cat_cond = self.cond_cross_norm(img_cat_cond, original_cond.unsqueeze(1))
                    else:
                        img_cat_cond = self.cond_cross_norm(img_cat_cond.unsqueeze(1), original_cond)
                        if len(img_cat_cond.shape) == 4:
                            img_cat_cond = img_cat_cond.squeeze(1)
                        img_cat_cond = rearrange(img_cat_cond, 'B (T H W) D -> B T D H W', T=T, H=H_enc, W=W_enc)
                else:
                    img_cat_cond = rearrange(img_cat_cond, 'B (T H W) D -> B T D H W', T=T, H=H_enc, W=W_enc)

                if self.use_zero_conv_latent_input:
                    img_cat_cond = rearrange(img_cat_cond, 'B T C H W -> B C T H W')
                    img_cat_cond = self.multi_cond_in_projection(img_cat_cond)
                    img_cat_cond = rearrange(img_cat_cond, 'B C T H W -> B T C H W')

                    img_cat_cond = original_cond.unsqueeze(1) + img_cat_cond
            else:
                img_cat_cond = rearrange(img_cat_cond, 'B (T H W) D -> B T D H W', T=T, H=H_enc, W=W_enc)

            img_cat_cond = rearrange(img_cat_cond, 'B T D H W -> B D T H W')
        else:
            img_cat_cond = z[torch.arange(batch_size, device=device), :, cond_frame_index, :, :]

        ########################################### only change here, add multi condition input ###########################################
        ###################################################################################################################################
        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))

        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        if not hasattr(self, "null_prompt"):
            self.null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, self.null_prompt, cond_emb.detach())
        

        ###################################################################################################################################
        ########################################### only change here, add multi condition input ###########################################
        batch_size = x.shape[0]
        img = x[torch.arange(batch_size, device=device), :, cond_frame_index, ...]
        if self.use_semantic_branch and cond_frames is not None:
            num_cond_frames = cond_frames.shape[1]
            img = x[torch.arange(batch_size, device=device), :, cond_frame_index, ...]
            img = torch.concatenate((img.unsqueeze(1), cond_frames), dim=1)
            img = input_mask.unsqueeze(-1) * img
            img = rearrange(img, "b t c h w -> (b t) c h w")
        else:
            img = input_mask * img
        ########################################### only change here, add multi condition input ###########################################
        ###################################################################################################################################

        ## img: b c h w
        ##-- CLIP Embedding --##
        
        img_emb = self.embedder(img)  ## b l c
        img_emb = self.image_proj_model(img_emb)
        ###################################################################################################################################
        ########################################### only change here, add multi condition input ###########################################
        
        if self.use_semantic_branch and 'cond_frames' in batch and batch['cond_frames'] is not None:
            img_emb = img_emb.view(batch_size, num_cond_frames+1, img_emb.shape[-2], img_emb.shape[-1])
            if self.multi_cond_strategy == 'pose_agent_enc':
                pose_features = camera_condition_kwargs['camera_condition']['pluker_embedding_features']
                self.multi_cond_func(
                    context = img_emb,
                    pose_embedding = pose_features,
                    attn_mask =  None # TODO: Implement epipolar masking
                )
                img_emb = rearrange(img_emb, "B C N D -> B (C N) D")
            else:
                img_emb = self.multi_cond_func(img_emb)
        ########################################### only change here, add multi condition input ###########################################
        ###################################################################################################################################
        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            elif self.multi_cond_strategy != 'token_concat_latent_epipolar':
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=T)

            cond["c_concat"] = [img_cat_cond]  # b c t h w 
            cond["c_cond_frame_index"] = cond_frame_index
            cond["origin_z_0"] = z.clone()
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)]  ## concat in the seq_len dim

        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')

            
        cond.update(camera_condition_kwargs)

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            out.append(fs)
        if return_cond_frame_index:
            out.append(cond_frame_index)
        if return_cond_frame:
            out.append(x[torch.arange(batch_size, device=device), :, cond_frame_index-T, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        if return_camera_data:
            camera_data = batch.get('camera_data', None)
            out.append(camera_data)

        if return_video_path:
            out.append(batch['video_path'])

        if return_depth_scale:
            out.append(depth_scale)
        return out

    def compute_conditional_epipolar_mask(self, batch, H, W, downsample_factor: int = 8, return_squeezed_tokens: bool = False, cond_frame_indices: Tensor = None):
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
            w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
            w2c_RT_4x4_cond = super().get_input(batch, 'RT_cond').float()  # b, c, 4, 4
            
            c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
            c2w_RT_4x4_cond = w2c_RT_4x4_cond.inverse()  # w2c --> c2w
            if cond_frame_indices is not None:
                c2w_RT_4x4_cond = torch.cat((c2w_RT_4x4[torch.arange(len(cond_frame_indices)), cond_frame_indices].unsqueeze(1), c2w_RT_4x4_cond), dim=1)

            relative_c2w_RT_4x4 = self.get_pairwise_relative_pose(c2w_RT_4x4_cond, c2w_RT_4x4)
            relative_c2w_RT_4x4 = rearrange(relative_c2w_RT_4x4, 'B T C H W -> B C T H W')

            R = relative_c2w_RT_4x4[..., :3, :3]  # b,t,t,3,3
            t = relative_c2w_RT_4x4[..., :3, 3:4]  # b,t,t,3,1

            C = R.shape[2]
            T = R.shape[1]
            K = repeat(camera_intrinsics_3x3, 'B T H W -> B (T C) H W', C=C)
            R = rearrange(R, 'B T C H W -> B (T C) H W')
            t = rearrange(t, 'B T C H W -> B (T C) H W')
            F = self.get_fundamental_matrix(K, R, t)
            F = rearrange(F, 'B (T C) H W -> B T C H W', C=C)
            
            T = c2w_RT_4x4.shape[1]
            epipolar_mask = self.get_epipolar_mask(F, T, H // downsample_factor, W // downsample_factor, downsample_factor, return_squeezed_tokens)

        return epipolar_mask



    def get_batch_input_camera_condition_process(self, batch, x, cond_frame_index, trace_scale_factor, rand_cond_frame, *args, **kwargs):
        return_log = {}
        return_kwargs = {}
        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            
            camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
            w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
            c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
            B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

            relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
            relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

            if self.epipolar_config is not None and not self.epipolar_config.is_3d_full_attn:
                relative_c2w_RT_4x4_pairs = self.get_relative_c2w_RT_pairs(relative_c2w_RT_4x4)  # b,t,t,4,4
                R = relative_c2w_RT_4x4_pairs[..., :3, :3]  # b,t,t,3,3
                t = relative_c2w_RT_4x4_pairs[..., :3, 3:4]  # b,t,t,3,1

                if self.epipolar_config.add_small_perturbation_on_zero_T:
                    t = self.add_small_perturbation(t, epsilon=1e-6)

                K = camera_intrinsics_3x3.unsqueeze(1)


                F = self.get_fundamental_matrix(K, R, t)
                
                sample_locs_dict = {d: self.get_epipolar_mask(F, T, H // d, W // d, d) for d in [int(8 * ds) for ds in self.epipolar_config.attention_resolution]}
            else:
                sample_locs_dict = None

        if self.pose_encoder is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                pluker_embedding = self.ray_condition(camera_intrinsics_3x3, relative_c2w_RT_4x4, H, W, device, flip_flag=None)  # b, 6, t, H, W

            pluker_embedding_features = self.pose_encoder(pluker_embedding)  # bf c h w
            pluker_embedding_features = [rearrange(_, '(b f) c h w -> b c f h w', b=batch_size) for _ in pluker_embedding_features]
        else:
            pluker_embedding_features = None

        return_kwargs["camera_condition"] = {
            "pluker_embedding_features": pluker_embedding_features,
            "sample_locs_dict": sample_locs_dict,
            "cond_frame_index": cond_frame_index,
            "add_type": self.add_type,
        }

        return return_log, return_kwargs
    
    def configure_optimizers(self):
        """
            Overwrites setup from base model
        """
        lr = self.learning_rate
        weight_decay = self.weight_decay

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

        if self.pose_encoder is not None and self.pose_encoder_trainable:
            params_pose_encoder = [p for p in self.pose_encoder.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_pose_encoder)}] Paramters for pose_encoder.")
            params.extend(params_pose_encoder)

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))
        
        if self.pose_guided_cond_trainable:
            mainlogger.info(f"@Training [{len(list(self.pose_agent_enc.parameters()))}] Paramters for pose-guided conditioner.")
            params.extend(list(self.pose_agent_enc.parameters()))
        
        if self.multi_cond_adaptor_trainable:
            mainlogger.info(f"@Training [{len(list(self.multi_cond_latent_adaptor.parameters()))}] Paramters for multi-condition adaptor.")
            params.extend(list(self.multi_cond_latent_adaptor.parameters()))

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        
        if self.use_zero_conv_latent_input:
            params.extend(list(self.multi_cond_in_projection.parameters()))

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer
    

    def _get_multi_cond_func(self, pose_guided_cond_config=None, dim: int = 1, *args, **kwargs):
        if self.multi_cond_strategy == 'max':
            return lambda x: torch.max(x, dim=dim).values
        elif self.multi_cond_strategy == 'avg':
            return lambda x: torch.mean(x, dim=dim)
        elif self.multi_cond_strategy in ['token_concat', 'token_concat_latent', 'token_concat_latent_epipolar']:
            if self.multi_cond_strategy in ['token_concat_latent', 'token_concat_latent_epipolar']:
                self.multi_cond_latent_adaptor = instantiate_from_config(kwargs.get("multi_latent_adaptor"))
                if self.epipolar_mask_freeze_steps is not None and self.epipolar_mask_freeze_steps > 0:
                    self.multi_cond_latent_adaptor.use_mask = False
                for _name, param in self.multi_cond_latent_adaptor.named_parameters():
                    param.requires_grad = self.multi_cond_adaptor_trainable
            return lambda x: rearrange(x, 'b t n d -> b (t n) d')
        else:
            logging.warning(f"Invalid multi_cond_strategy: {self.multi_cond_strategy}. Keeping input unchanged.")
            return lambda x: x

    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=50,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        mask=None,
        sampled_img_num=1,
        enable_camera_condition=True,
        trace_scale_factor=1.0,
        cond_frame_index=None,
        **kwargs,
    ):
        # Overwrites functions in parent class
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        for key in batch.keys():
            if batch[key] is None:
                continue
            if isinstance(batch[key], str) \
                or isinstance(batch[key], float) \
                or isinstance(batch[key], int):
                continue
            elif isinstance(batch[key], list) and len(batch[key]) < sampled_img_num:
                continue
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_frame_index, cond_x, x, camera_data, video_path, depth_scale = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
            return_fs=True,
            return_cond_frame_index=True,
            return_cond_frame=True,
            rand_cond_frame=False,
            enable_camera_condition=enable_camera_condition,
            return_original_input=True,
            return_camera_data=True,
            return_video_path=True,
            return_depth_scale=True,
            trace_scale_factor=trace_scale_factor,
            cond_frame_index=cond_frame_index,
        )
        N = xrec.shape[0]
        log["depth_scale"] = depth_scale
        log["camera_data"] = camera_data
        log["video_path"] = video_path
        log["gt_video"] = x
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        if 'cond_frames' in batch.keys():
            log["cond_frames"] = batch["cond_frames"]
        xc_with_fs = []
        
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)
                elif self.uncond_type == "negative_prompt":
                    prompts = N * [kwargs["negative_prompt"]]
                    uc_prompt = self.get_learned_conditioning(prompts)

                img = torch.zeros_like(xrec[:, :, 0])  ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img)  ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            pre_process_log, pre_process_kwargs = self.log_images_sample_log_pre_process(
                batch, z, x, cond_frame_index, trace_scale_factor, **kwargs
            )
            log.update(pre_process_log)
            kwargs.update(pre_process_kwargs)

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z,
                                                         enable_camera_condition=enable_camera_condition, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            log.update(self.log_images_sample_log_post_process(x_samples, **pre_process_log))

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.multi_cond_strategy == 'token_concat_latent_epipolar':
            if self.epipolar_mask_freeze_steps is not None and self.multi_cond_latent_adaptor.use_mask == False and self.global_step >= self.epipolar_mask_freeze_steps:
                mainlogger.info("Unlock epipolar mask usage in multi-latent adaptor.")
                self.multi_cond_latent_adaptor.use_mask = True
        super().on_train_batch_end(*args, **kwargs)


    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, *add_args = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, **kwargs)
        return loss, loss_dict
    
    def forward(self, x, c, **kwargs):
        if 't' in kwargs:
            t = kwargs['t']
            del kwargs['t']
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)
    
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == 'l2_log':
            weight = torch.log10(torch.arange(2, pred.shape[2]+2, device=pred.device))
            weight = (weight / torch.sum(weight))
            weight = rearrange(weight, "N -> 1 1 N 1 1")
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
            loss = weight * loss
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    @torch.no_grad()
    def sample_down_emb_unet(self, x, cond, t, noise=None, **kwargs):
        
        # From: LatentDiffusion->p_losses(...)
        if self.noise_strength > 0:
            b, c, f, _, _ = x.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x.device)
            noise = default(noise, lambda: torch.randn_like(x) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        # From: LatentDiffusion->apply_model(...)
        c_concat = cond['c_concat']
        c_crossattn = cond['c_crossattn']
        # From: DiffusionWrapper->forward(...)
        xc = torch.cat([x_noisy] + c_concat, dim=1)
        cc = torch.cat(c_crossattn, 1)

        skip_connections = self.model.diffusion_model.downwards_embed(xc, t, context=cc, **cond, **kwargs)
        skip_connections = [s.detach() for s in skip_connections[::-1]]
        return skip_connections


