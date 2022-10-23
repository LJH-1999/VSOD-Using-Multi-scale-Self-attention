import torch.nn as nn
import torch
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

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
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale



        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio == 4:
                self.sr1 = nn.Conv3d(dim, dim, kernel_size=(5, 4, 4), padding=(8, 0, 0), stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv3d(dim, dim, kernel_size=(1, 2, 2), padding=(2, 0, 0), stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio == 2:
                self.sr1 = nn.Conv3d(dim, dim, kernel_size=(1, 2, 2), padding=(2, 0, 0), stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1, stride=1, groups=dim // 2)
            self.local_conv2 = nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1, stride=1, groups=dim // 2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, stride=1, groups=dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, num_frame, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # bhnd
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, num_frame, H, W)
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]  # B head N C
            k2, v2 = kv2[0], kv2[1]
            attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, num_frame, H // self.sr_ratio, W // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2) #bhnd
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
            attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, num_frame, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2) #bhnd
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)

            x = torch.cat([x1, x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, num_frame, H, W)).view(B,C,N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, group=5, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth // 2):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention2(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        for _ in range(depth // 2, depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.group = group

    def forward(self, x):
        bs_gp, dim, wid, hei = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        bs = bs_gp // self.group
        gp = self.group
        x = x.reshape(bs, gp, dim, wid, hei)
        x = x.permute(0, 1, 3, 4, 2).reshape(bs, gp * wid * hei, dim)

        for attn, ff in self.layers[:2]:
            x = attn(x, num_frame=gp, H=hei, W=wid)
            x = ff(x)
        for attn, ff in self.layers[2:]:
            x = attn(x)
            x = ff(x)

        x = x.reshape(bs, gp, wid, hei, dim).permute(0, 1, 4, 2, 3).reshape(bs_gp, dim, wid, hei)

        return x



f = nn.Conv3d(512, 512, kernel_size=(1, 1, 1), stride=1)
x = torch.ones(size=(8, 512, 5, 56, 56))
x = f(x)
print(x.shape)

attention = Attention2(dim=512, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=4)
x = attention.forward(x=torch.randn(size=(8,5*14*14,512)), num_frame=5, H=14, W=14)
print(x.shape)

transformer = Transformer(512,4,4,782,group=5,dropout=0.)
x = transformer.forward(x=torch.randn(size=(40,512,28,28)))
print(x.shape)
