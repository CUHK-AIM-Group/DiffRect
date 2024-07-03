# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

from .guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.resample import UniformSampler

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),)
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.temb_proj = torch.nn.Linear(512,
                                         out_channels)

    def forward(self, x, temb):
        x = self.conv0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv1(x)
        return x


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, temb):
        x = self.maxpool(x)
        x = self.conv(x, temb)
        return x


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2, temb):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        # print(x1.shape, x2.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, temb)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0]) # 16
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1]) # 32
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2]) # 64
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3]) # 128
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4]) # 256

    def forward(self, x, temb, embeddings=None):
        x0 = self.in_conv(x, temb)
        if embeddings is not None:
            x0 = x0 + embeddings[0]
        x1 = self.down1(x0, temb)
        if embeddings is not None:
            x1 = x1 + embeddings[1]
        x2 = self.down2(x1, temb)
        if embeddings is not None:
            x2 = x2 + embeddings[2]
        x3 = self.down3(x2, temb)
        if embeddings is not None:
            x3 = x3 + embeddings[3]
        x4 = self.down4(x3, temb)
        if embeddings is not None:
            x4 = x4 + embeddings[4]
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        # self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.out_chns = self.params['out_chns']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.out_chns,
                                  kernel_size=3, padding=1)

    def forward(self, feature, temb, out_multi=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        outs = []
        x = self.up1(x4, x3, temb)
        outs.append(x)
        x = self.up2(x, x2, temb)
        outs.append(x)
        x = self.up3(x, x1, temb)
        outs.append(x)
        x = self.up4(x, x0, temb)
        output = self.out_conv(x)
        outs.append(output)
        if out_multi:
            return outs
        return output

  
      
class DeUNet(nn.Module):
    def __init__(self):
        super(DeUNet, self).__init__()
        self.ft_chns = [256, 384, 512]
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], 0.0) # 32
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], 0.0) # 64
        self.up1 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
        
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
    def forward(self, x, temb, embeddings=None):
        
        temb = get_timestep_embedding(temb, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        B, C, H, W = x.shape
        x0 = x
        if len(embeddings) == 1:
            x0 = x0 + embeddings[0]
        x1 = self.down1(x, temb)
        if len(embeddings) > 1:
            if embeddings is not None:
                x1 = x1 + embeddings[-2]
        x2 = self.down2(x1, temb)
        if len(embeddings) > 1:
            if embeddings is not None:
                x2 = x2 + embeddings[-1]
        x = self.up1(x2, x1, temb)
        x = self.up2(x, x0, temb)
        assert x.shape == (B, C, H, W)
        return x
    
class DiffUNet(nn.Module):
    def __init__(self, ts=1000, ts_sample=10, ldm_sch='linear') -> None:
        super().__init__()
        
        self.model = DeUNet()

        betas = get_named_beta_schedule(ldm_sch, ts)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(ts, [ts]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(ts, [ts_sample]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(ts)


    def forward(self, x=None, pred_type=None, step=None, embeddings=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            return self.model(x, temb=step, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            if len(embeddings) == 1:
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, embeddings[0].shape, model_kwargs={"embeddings": embeddings})
            else:
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, embeddings[-3].shape, model_kwargs={"embeddings": embeddings[-2:]})
            sample_out = sample_out["pred_xstart"]
            return sample_out

        
class UNet_LDMV2(nn.Module):
    def __init__(self, in_chns, class_num, out_chns, ldm_method='adaptor', ldm_beta_sch='linear', ts=1000, ts_sample=10):
        super(UNet_LDMV2, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'out_chns': out_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns - 3, # in chns - Maskige
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'out_chns': out_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
        self.encoder = Encoder(params)
        self.embedder = Encoder(params2)
        self.decoder = Decoder(params)

        self.deunet = DiffUNet(ts=ts, ts_sample=ts_sample, ldm_sch=ldm_beta_sch)
        self.de_loss = nn.MSELoss()
        
        self.ldm_method = ldm_method

        if ldm_method == 'adaptor':
            self.adaptor = ConvBlock(512, 256, 0.0)

    def get_lat_loss(self, pred, gt):
        return self.de_loss(pred, gt)

    def forward(self, x, t, image=None, training=True, good=None, save_feature_iter=False, iter_num=-1):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            img_embeddings = self.embedder(image, temb)
            x = torch.cat([image, x], dim=1)
        else:
            img_embeddings = None
            # raise NotImplementedError
        feature = self.encoder(x, temb, img_embeddings)
        if training:
            good = torch.cat([image, good], dim=1)
            feature_good = self.encoder(good, temb, img_embeddings)[-1].detach()
            x_start = feature_good
            x_t, t_lat, noise = self.deunet(x=x_start, pred_type="q_sample") # add noise to label
            pred_xstart = self.deunet(x=x_t, step=t_lat, pred_type="denoise", embeddings=[feature[-1]])
            if self.ldm_method == 'adaptor':
                feat_ref = torch.cat([feature[-1], pred_xstart], dim=1)
            elif self.ldm_method == 'add':
                feat_ref = feature[-1] + pred_xstart
            elif self.ldm_method == 'replace':
                feat_ref = pred_xstart
            lat_loss = self.get_lat_loss(pred_xstart, x_start)
        else:
            assert good is None
            sample_xstart = self.deunet(pred_type="ddim_sample", embeddings=[feature[-1]])
            if self.ldm_method == 'adaptor':
                feat_ref = torch.cat([feature[-1], sample_xstart], dim=1)
            elif self.ldm_method == 'add':
                feat_ref = feature[-1] + sample_xstart
            elif self.ldm_method == 'replace':
                feat_ref = sample_xstart

        if self.ldm_method == 'adaptor':
            feature[-1] = self.adaptor(feat_ref, temb)
        else:
            feature[-1] = feat_ref
        output = self.decoder(feature, temb, out_multi=True)
        output = output[-1]

        if training:
            return lat_loss, output
        else:
            return output
        
    
if __name__ == '__main__':
    model = UNet_LDMV2(4, 2, 2)
    x = torch.rand(1, 3, 256, 256)
    image = torch.rand(1, 1, 256, 256)
    t = torch.rand(1)
    output = model(x, t, image, True)
    print(output[0], output[1].shape)
    output = model(x, t, image, False)
    print(output.shape)
