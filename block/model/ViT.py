## Implementation slightly adapted from: https://git-core.megvii-inc.com/sunboran/neuracrypt_era/-/tree/master/models
import torch
from torch import nn, einsum
import torch.nn.functional as F

from . import model_utils
# import model_utils
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary 

class Encoder(nn.Module):
    def __init__(self, depth, img_size, kernel_size, in_channels, hidden_channels, channel_dilation=1, private=True):
        super(Encoder, self).__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_channels * channel_dilation, kernel_size, stride=kernel_size),
            nn.PReLU()
        ]
        # output size: (img_size // kernel_size, img_size // kernel_size), e.g., (112 // 16, 112 // 16)
        for _ in range(depth):
            layers.extend([
                nn.Conv2d(hidden_channels * channel_dilation, hidden_channels * channel_dilation, kernel_size=1, stride=1),
                nn.BatchNorm2d(hidden_channels * channel_dilation, track_running_stats=False),
                nn.PReLU()
            ])
        # output size: equal to input_size
        self.img_size = img_size
        
        self.n_patches = (img_size // kernel_size) ** 2  # 7^2
        self.hidden_channels = hidden_channels
        self.channel_dilation = channel_dilation

        self.image_encoder = nn.Sequential(*layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, hidden_channels * channel_dilation), requires_grad=False)
        self.private = private
        self.mixer = nn.Sequential(*[
            nn.PReLU(),
            nn.Linear(hidden_channels * channel_dilation, hidden_channels)
        ])

    def forward(self, x):
        encoded = self.image_encoder(x)
        # out: B * C * (H // kernel_size) * (W // kernel_size)

        B, C, H, W = encoded.size()
        encoded = encoded.view([B, -1, H * W]).transpose(1, 2)

        if self.private:
            encoded += self.pos_embedding
        # out: B * [(H // kernel_size) * (W // kernel_size)] * C

        encoded = self.mixer(encoded)
        # out: B * [(H // kernel_size) * (W // kernel_size)] * C
        if self.private:
            shuffled = torch.zeros_like(encoded)
            for i in range(B):
                idx = torch.randperm(H * W)
                for j, k in enumerate(idx):
                    shuffled[i, j] = encoded[i, k]
            encoded = shuffled

        return encoded

    def img_transform(self, img):
        '''H*W*3*uint8(0\~255) -> 3*112*112*float32(-1\~1)'''
        return model_utils.resize_norm_transpose(img, self.img_size)

    def refresh(self):
        hidden_channels = self.hidden_channels
        channel_dilation = self.channel_dilation
        self.pos_embedding.data = nn.Parameter(torch.randn(1, self.n_patches, hidden_channels * channel_dilation), requires_grad=False).data.cuda()
        for layer in self.image_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.mixer:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention 
        # 这段代码会导致程序被杀，查看内存占用发现，到此处内存飙升至溢出。
        # 而且ViT-pytorch的github repo的实现中也没有这段代码，暂时注释掉，等日后搞清写这段的目的
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        # 6 4 8 64 8 0
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # print('dida')
            x = attn(x)
            # print(x)
            x = ff(x)
        return x

class DeepViT(nn.Module):
    def __init__(self,feat_size:int, dim:int, depth:int, heads:int, mlp_dim:int, 
    pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        '''
        Input: tensor(B, num_patches, Channel)

        Params
        -----
        (int) dim: the input feature map channel
        (int) depth:  depth of transformer
        (int) heads: number of heads of
        (int) mlp_dim: hidden dims in mlp of transformer
        (int) heads: heads of attention, default: 8
        '''
        super().__init__()
        self.feat_size = feat_size
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_size // patch_size) ** 2
        # patch_dim = channels * patch_size ** 2

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # nn.Linear(patch_dim, dim),
        # )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) 

        # neuraCrypt使用的ViT部分，从此处开始
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # 6 4 8 64 8 0

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.feat_size)
        )

    def forward(self, x):
        b, n, c = x.shape
        # print(b, n, c) # batch_size, n_patches, patch_channels = (32, 49, 2048)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('cat_cls',x.shape)
        x = self.dropout(x)
        # print('dropout', x.shape)
        x = self.transformer(x)
        # print('tfm', x.shape)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def forward_transformer(self, x):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        return x


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.getcwd())) 
    sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) 
    # 找到上两级目录“ ..XX/privdl/”
    import block
    from torch.utils.data.dataloader import DataLoader

    # block_test = 'enc'
    block_test = 'vit'
    ds = block.dataset.hubble.xnn_paper.celeba()
    if block_test == 'enc':
        encoder = Encoder(3, 256, 4, 3, 6, 1)
        summary(encoder, input_size=(3, 256, 256), device="cpu")
        print(type(ds[0][0]), ds[0][0].shape)
        img = torch.Tensor(ds[0][0].transpose((2,0,1)))
        img = img.unsqueeze(dim=0)
        output = encoder(img)
        print(output.size())
    else: 
        cls_num = ds.class_num()
        dl = DataLoader(ds, 1, True)
        encoder = Encoder(depth=3, img_size=256, kernel_size=4, in_channels=3, hidden_channels=6, channel_dilation=1)
        VIT = DeepViT(num_classes=cls_num, dim=6, depth=4, heads=8, mlp_dim=8)
        summary(VIT, input_size=(4096, 6), device="cpu")
        for batch in dl:
            img = rearrange(batch[0], 'b1 h w c -> b1 c h w')
            # print(type(img), img.shape)
            output = encoder(img.float()) # output:(batch, num_patches, hidden_channels  )
            print(output.size())
            '''
            dim should be the same as output channel
            '''
            output = VIT(output)

            print('output of ViT:', output.size())
