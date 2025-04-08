
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Literal
from functools import partial
from einops import repeat, rearrange

from lvdm.common import checkpoint
from lvdm.modules.encoders.resampler import FeedForward, PerceiverAttention
from lvdm.models.utils_diffusion import timestep_embedding
from model.modules.epipolar import EpipolarCrossAttention

class TransposedConvolution(nn.Module):
    """
        More efficient transposed convolution implementation that also mitigates artifacts:
            https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TransposedConvolution, self).__init__()
        self.scale_factor = stride
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

class MultiLatentEpipolarAdaptor(nn.Module):

    def __init__(
        self,
        query_dim=512,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=1024,
        output_queries=None,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        num_register_tokens=2,
        use_mask = True,
        checkpoint=False,
        video_length=None, # using frame-wise version or not
        use_plucker_embedding=False,
        allow_plucker_embedding_param=False,
        context_positional_encoding=False,
        context_positional_encoding_dim = None,
        timestep_embedding_type: Literal['none', 'sinusoidal', 'sinusoidal_embedded']='none',
        timestep_embedding_dim=32,
        plucker_embedding_dim=320,
        plucker_input_strategy: Literal["concat", "add"] = "add",
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries 
        self.video_length = video_length if video_length is not None else 16
        self.use_mask = use_mask
        self.checkpoint = checkpoint
        output_queries = output_queries if output_queries is not None else num_queries
        self.use_plucker_embedding = use_plucker_embedding
        self.timestep_embedding_type = timestep_embedding_type
        self.timestep_embedding_func = None
        self.context_positional_encoding = context_positional_encoding
        self.context_positional_encoding_dim = context_positional_encoding_dim or embedding_dim if context_positional_encoding else 0
        self.timestep_embedding_dim = timestep_embedding_dim if self.timestep_embedding_type.split('_')[-1] == "embedded" else embedding_dim
        if self.timestep_embedding_type == "sinusoidal_embedded":
            self.timestep_embedding_func = nn.Sequential(
                nn.Linear(timestep_embedding_dim, query_dim),
                nn.SiLU(),
                nn.Linear(query_dim, query_dim),
            )

        ## <num_queries> queries for each frame
        num_queries_per_frame = num_queries
        if video_length is not None: 
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, query_dim) / query_dim**0.5)
        self.proj_in = nn.Linear(embedding_dim + self.context_positional_encoding_dim, query_dim)
        self.proj_out = nn.Linear(query_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.plucker_in = None
        self.plucker_input_strategy = plucker_input_strategy
        if use_plucker_embedding or allow_plucker_embedding_param:
            self.plucker_in = nn.Linear(plucker_embedding_dim, query_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        EpipolarCrossAttention(
                            query_dim = 2*query_dim if self.plucker_input_strategy == "concat" else query_dim,
                            context_dim = query_dim,
                            out_dim = query_dim,
                            num_register_tokens = num_register_tokens
                        ),
                        FeedForward(dim=query_dim, mult=ff_mult),
                    ]
                )
            )


        upscale_blocks = []
        self.upscaler = None
        if output_queries != num_queries_per_frame:
            assert output_queries % num_queries_per_frame == 0, "output_queries must be a whole-numbered multiple of num_queries"

            upscale_factor = int(output_queries // num_queries_per_frame)
            upsample_kernel_size = 3
            upsample_padding = int(upsample_kernel_size // 2)

            for i in range(upscale_factor - 1):
                block = nn.Sequential(
                    TransposedConvolution(
                        in_channels = query_dim,
                        out_channels = query_dim,
                        kernel_size = upsample_kernel_size,
                        padding = upsample_padding,
                        stride = 2
                    )
                )
                upscale_blocks.append(block)

            self.upscaler = nn.Sequential(*upscale_blocks)

    def forward(self, x, mask=None, plucker_embedding_features=None):
        forward_method = partial(self._forward, mask=mask, plucker_embedding_features=plucker_embedding_features)
        return checkpoint(forward_method, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x, mask=None, plucker_embedding_features=None):
        """
        
        """
        latents = self.latents.repeat(x.size(0), 1, 1) ## B (T L) C
        if self.context_positional_encoding:
            n = x.shape[1] // self.num_queries
            pos_enc = timestep_embedding(torch.arange(n), self.num_queries).to(x.device)
            pos_enc = repeat(pos_enc, "N M -> N M D", D=self.context_positional_encoding_dim)
            pos_enc = rearrange(pos_enc, "N M D -> 1 (N M) D")
            pos_enc = repeat(pos_enc, "1 N D -> B N D", B=x.shape[0])
            x = torch.concatenate((x, pos_enc), dim=-1)
        x = self.proj_in(x)
        if plucker_embedding_features is not None:
            plucker_embedding_features = self.plucker_in(plucker_embedding_features)
            

        if not self.use_mask:
            mask = None
        
        for attn, ff in self.layers:
            if plucker_embedding_features is not None:
                if self.plucker_input_strategy == "concat":
                    input_latents = torch.concatenate((latents, plucker_embedding_features), dim=-1)
                elif self.plucker_input_strategy == "add":
                    input_latents = latents + plucker_embedding_features
            else:
                input_latents = latents
            latents = attn(input_latents, x, mask) + latents
            latents = ff(latents) + latents

        if self.timestep_embedding_type != 'none':
            t_emb = timestep_embedding(torch.arange(self.video_length), self.timestep_embedding_dim, repeat_only=False).to(dtype=latents.dtype, device=latents.device)
            t_emb = self.timestep_embedding_func(t_emb) if self.timestep_embedding_func is not None else t_emb
            num_repeats = latents.shape[1] // self.video_length
            t_emb = repeat(t_emb.view(1, *t_emb.shape, 1), "1 T C 1 -> B C T L", B=latents.shape[0], L = num_repeats)
            t_emb = rearrange(t_emb, "B C T L -> B (T L) C")
            latents = latents + t_emb
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents) # B L C or B (T L) C

        return latents


class MultiLatentAdaptor(nn.Module):
    """
        A query-transformer architecture to encode multiple-latent embeddings into a single latent embedding.
    """

    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        video_length=None, # using frame-wise version or not
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries 
        self.video_length = video_length

        ## <num_queries> queries for each frame
        if video_length is not None: 
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1) ## B (T L) C
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents) # B L C or B (T L) C

        return latents