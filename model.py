import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    frequencies = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimeEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class TimeAwareFusion(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.fusion_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * dim), 
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv3d(dim, dim, kernel_size=1) 

    def forward(self, x_a, x_b, t_emb):
        gates = self.fusion_gate(t_emb)
        w_a, w_b = gates.chunk(2, dim=1)
        
        w_a = w_a.view(x_a.shape[0], x_a.shape[1], 1, 1, 1)
        w_b = w_b.view(x_b.shape[0], x_b.shape[1], 1, 1, 1)
        
        x_fused = w_a * x_a + w_b * x_b
        
        return self.proj(x_fused)

class TAED(nn.Module):
    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        use_checkpoint: bool = False,
        norm_name: str = "instance",
    ):
        super().__init__()
        
        self.feature_size = feature_size
        
        self.time_embed_dim = feature_size * 4 
        self.t_embedder = TimeEmbedder(feature_size, self.time_embed_dim)

        self.swin_t1 = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        self.swin_flair = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        self.fusion_0 = TimeAwareFusion(feature_size * 1, self.time_embed_dim)
        self.fusion_1 = TimeAwareFusion(feature_size * 2, self.time_embed_dim)
        self.fusion_2 = TimeAwareFusion(feature_size * 4, self.time_embed_dim)
        self.fusion_3 = TimeAwareFusion(feature_size * 8, self.time_embed_dim)
        self.fusion_4 = TimeAwareFusion(feature_size * 16, self.time_embed_dim)

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3, 
            in_channels=feature_size * 16, 
            out_channels=feature_size * 8, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3, 
            in_channels=feature_size * 8, 
            out_channels=feature_size * 4, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3, 
            in_channels=feature_size * 4, 
            out_channels=feature_size * 2, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3, 
            in_channels=feature_size * 2, 
            out_channels=feature_size, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3, 
            in_channels=feature_size, 
            out_channels=feature_size, 
            kernel_size=3, 
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        
        self.out = UnetOutBlock(3, feature_size, out_channels)
        
        self.encoder1 = UnetrBasicBlock(3, in_channels, feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder2 = UnetrBasicBlock(3, feature_size, feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder3 = UnetrBasicBlock(3, 2 * feature_size, 2 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder4 = UnetrBasicBlock(3, 4 * feature_size, 4 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder10 = UnetrBasicBlock(3, 16 * feature_size, 16 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)

        self.dec_time_proj_4 = nn.Linear(self.time_embed_dim, feature_size * 8 * 2) 
        self.dec_time_proj_3 = nn.Linear(self.time_embed_dim, feature_size * 4 * 2)
        self.dec_time_proj_2 = nn.Linear(self.time_embed_dim, feature_size * 2 * 2)
        self.dec_time_proj_1 = nn.Linear(self.time_embed_dim, feature_size * 1 * 2)

    
    def forward_encoder(self, x, swin_model):
        return swin_model(x, normalize=True)

    def apply_time_modulation(self, x, t_emb, proj_layer):
        params = proj_layer(t_emb)
        scale, shift = params.chunk(2, dim=1)
        scale = scale.view(x.shape[0], -1, 1, 1, 1)
        shift = shift.view(x.shape[0], -1, 1, 1, 1)
        return x * (1 + scale) + shift

    def forward(self, x_t1, x_flair, timesteps):
        t_emb = self.t_embedder(timesteps) 

        feats_t1 = self.forward_encoder(x_t1, self.swin_t1)
        feats_flair = self.forward_encoder(x_flair, self.swin_flair)
        
        f_0 = self.fusion_0(feats_t1[0], feats_flair[0], t_emb)
        f_1 = self.fusion_1(feats_t1[1], feats_flair[1], t_emb)
        f_2 = self.fusion_2(feats_t1[2], feats_flair[2], t_emb)
        f_3 = self.fusion_3(feats_t1[3], feats_flair[3], t_emb)
        f_4 = self.fusion_4(feats_t1[4], feats_flair[4], t_emb)

        enc0 = self.encoder1(x_t1) 
        enc1 = self.encoder2(f_0)
        enc2 = self.encoder3(f_1)
        enc3 = self.encoder4(f_2)
        
        dec4 = self.encoder10(f_4)

        dec3 = self.decoder5(dec4, f_3) 
        dec3 = self.apply_time_modulation(dec3, t_emb, self.dec_time_proj_4)
        
        dec2 = self.decoder4(dec3, enc3)
        dec2 = self.apply_time_modulation(dec2, t_emb, self.dec_time_proj_3)
        
        dec1 = self.decoder3(dec2, enc2)
        dec1 = self.apply_time_modulation(dec1, t_emb, self.dec_time_proj_2)
        
        dec0 = self.decoder2(dec1, enc1)
        dec0 = self.apply_time_modulation(dec0, t_emb, self.dec_time_proj_1)
        
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        
        return logits
