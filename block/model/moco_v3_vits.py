# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg, checkpoint_seq, named_apply, get_init_weights_vit, Block, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

__all__ = [
    'vit_small', 
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.num_tokens = 1 # update 2022.11.16 by lkx
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# costumized model for XNN, InstaHide, etc.
class ViTMoCo_5and6(VisionTransformerMoCo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


class VisionTransformerMoCo_Head(VisionTransformerMoCo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # del self.norm
        del self.fc_norm
        del self.head 
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        # x = self.forward_features(x)
        # x = self.forward_head(x)
        x = self.norm(x)
        return x


class VisionTransformerMoCo_Tail(VisionTransformerMoCo):
    def __init__(self, debug_tail_pos=False,  **kwargs):
        self.debug_tail_pos = debug_tail_pos
        super().__init__(**kwargs)
        del self.norm_pre
        if self.debug_tail_pos:
            self.patch_embed.num_patches = 197
            self.cls_token = None
        else:
            del self.cls_token
            del self.pos_embed
            del self.pos_drop
        
        
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.debug_tail_pos:    
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def forward_features(self, x):
        # x = self.patch_embed(x)
        if self.debug_tail_pos: 
            x = self._pos_embed(x)
        # x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


class Block_no_pre_norm(Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.norm1

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(x)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformerMoCo_PrivEnc(VisionTransformerMoCo):
    def __init__(self, **kwargs):
        self.obf = kwargs.pop('obf', 'dim')
        super().__init__(**kwargs)

        num_patches = self.patch_embed.num_patches + self.num_prefix_tokens
        self.nc_pos_embed = nn.Parameter(torch.randn(1, num_patches, self.embed_dim) * .02)
        
        self.mixer = nn.Sequential(*[
            nn.PReLU(),
            nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim),
        ])

        self.rearr = Rearrange('b n d -> b d n')

        del self.fc_norm
        del self.head
    
    def forward(self, x):
        
        x = self.forward_features(x) # encode
        
        x = x + self.nc_pos_embed # add_pos
        
        encoded = self.mixer(x) # mixer

        # CNN:channel内shuffle, vit:patch内shuffle
        # encoded = self.rearr(encoded)
        # shuffled = torch.zeros_like(encoded)
        # for i in range(encoded.shape[0]):
        #     idx = torch.randperm(encoded.shape[1])
        #     for j, k in enumerate(idx):
        #         shuffled[i, j] = encoded[i, k]
        # encoded = self.rearr(shuffled)

        b, n, d = encoded.shape
        if self.obf == 'dim':
            idx = torch.stack([torch.randperm(d) for _ in range(b * n)], dim=0).view(b, n, d).to(encoded.device)
            encoded = encoded.gather(2, idx)
        elif self.obf == 'dim_equal': # 每个patch的shuffle相同
            idx = torch.randperm(d).to(encoded.device)
            # expand idx to [b, n, d]
            encoded = encoded.gather(2, idx.unsqueeze(0).unsqueeze(0).expand(b, n, d))
        elif self.obf == 'patch':
            idx = torch.stack([torch.randperm(n) for _ in range(b)], dim=0).to(encoded.device)
            encoded = encoded.gather(1, idx.unsqueeze(-1).expand(b, n, d))
        elif self.obf == 'no':
            pass
        
        return encoded

class VisionTransformerMoCo_CLF(VisionTransformerMoCo):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        del self.pos_embed
        del self.pos_drop
        
        
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0. 
        # trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def forward_features(self, x):
        tmp = self.cls_token.expand(x.shape[0], -1, -1)
        x[:, 0, :] = torch.squeeze(tmp, dim=1)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


#custom model for NC
def vit_base_privencoder(layer, **kwargs):
    model = VisionTransformerMoCo_PrivEnc(
        obf =kwargs.pop('obf', 'dim'), patch_size=16, embed_dim=768, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base_clf(layer, **kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo_CLF(
        num_classes=0, embed_dim=768, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True,  **kwargs)
    model.default_cfg = _cfg()
    return model

# custom model for XNN
def vit_base_5and6(layer, **kwargs):
    '''
    without MLP head
    '''
    # ViTMoCo_5and6
    model = ViTMoCo_5and6(
        patch_size=16, num_classes=0, embed_dim=768, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, fc_norm=False, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base_head(layer, **kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo_Head(
        patch_size=16, embed_dim=768, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base_tail(layer, debug_pos = False, **kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo_Tail(
        debug_tail_pos = debug_pos, num_classes=0, embed_dim=768, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, fc_norm=False,  **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base_tail_no_pre_norm(layer, **kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo_Tail(
        num_classes=0, embed_dim=768, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, fc_norm=False, block_fn=Block_no_pre_norm, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_small_head(layer, **kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo_Head(
        patch_size=16, embed_dim=384, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_small_tail(layer, **kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo_Tail(
        num_classes=0, embed_dim=384, depth=layer, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=True, fc_norm=False,  **kwargs)
    model.default_cfg = _cfg()
    return model


#custom model for InstaHide
def vit_base_nomlp(stop_grad=True ,**kwargs):
    '''
    without MLP head
    '''
    model = VisionTransformerMoCo(
        num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=stop_grad,  **kwargs)
    
    model.default_cfg = _cfg()
    return model