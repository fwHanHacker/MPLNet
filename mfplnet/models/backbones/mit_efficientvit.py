import math
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list or tuple or any, min_len: int = 1, idx: int = 0) -> tuple:
    x = val2list(x, min_len)
    return tuple(x[idx:] + x[:idx])

def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else sum(x[1:], x[0])

def build_kwargs_from_config(config: dict, target_class: type) -> dict:
    valid_args = set(target_class.__init__.__code__.co_varnames[1:])
    return {k: v for k, v in config.items() if k in valid_args}

def resize(
    x: torch.Tensor,
    size: Optional[tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    else:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)

def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "bn2d":
        return nn.BatchNorm2d(num_features)
    elif name == "ln2d":
        return nn.LayerNorm2d(num_features)
    elif name == "gn":
        return nn.GroupNorm(num_groups=num_features//8, num_channels=num_features)
    else:
        raise ValueError(f"Unsupported norm type: {name}")

def build_act(name: Optional[str], inplace: bool = True) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif name == "hswish":
        return nn.Hardswish(inplace=inplace)
    elif name == "hsigmoid":
        return nn.Hardsigmoid(inplace=inplace)
    elif name == "swish" or name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = get_norm(norm, out_channels)
        self.act = get_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


def get_norm(norm: Optional[str], num_features: int) -> Optional[nn.Module]:
    if norm is None:
        return None
    elif norm == "bn2d":
        return nn.BatchNorm2d(num_features)
    elif norm == "ln2d":
        return nn.LayerNorm2d(num_features)
    elif norm == "gn":
        return nn.GroupNorm(num_groups=num_features//8, num_channels=num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


def get_act(act_func: Optional[str]) -> Optional[nn.Module]:
    if act_func is None:
        return None
    elif act_func == "relu":
        return nn.ReLU(inplace=True)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=True)
    elif act_func == "hswish":
        return nn.Hardswish(inplace=True)
    elif act_func == "hsigmoid":
        return nn.Hardsigmoid(inplace=True)
    elif act_func == "swish":
        return nn.SiLU(inplace=True)
    elif act_func == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_func}")


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("swish", "swish", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("swish", None),
    ):
        super(FusedMBConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x

class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out

class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super(EfficientViTBlock, self).__init__()
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                ),
                IdentityLayer(),
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")

        if local_module == "MBConv":
            self.local_module = ResidualBlock(
                MBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        elif local_module == "GLUMBConv":
             class GLUMBConv(nn.Module):
                def __init__(
                    self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size=3,
                    stride=1,
                    mid_channels=None,
                    expand_ratio=6,
                    use_bias=False,
                    norm=(None, None, "ln2d"),
                    act_func=("silu", "silu", None),
                ):
                    super().__init__()
                    use_bias = val2tuple(use_bias, 3)
                    norm = val2tuple(norm, 3)
                    act_func = val2tuple(act_func, 3)

                    mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

                    self.glu_act = build_act(act_func[1], inplace=False)
                    self.inverted_conv = ConvLayer(
                        in_channels,
                        mid_channels * 2,
                        1,
                        use_bias=use_bias[0],
                        norm=norm[0],
                        act_func=act_func[0],
                    )
                    self.depth_conv = ConvLayer(
                        mid_channels * 2,
                        mid_channels * 2,
                        kernel_size,
                        stride=stride,
                        groups=mid_channels * 2,
                        use_bias=use_bias[1],
                        norm=norm[1],
                        act_func=None,
                    )
                    self.point_conv = ConvLayer(
                        mid_channels,
                        out_channels,
                        1,
                        use_bias=use_bias[2],
                        norm=norm[2],
                        act_func=act_func[2],
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.inverted_conv(x)
                    x = self.depth_conv(x)

                    x, gate = torch.chunk(x, 2, dim=1)
                    gate = self.glu_act(gate)
                    x = x * gate

                    x = self.point_conv(x)
                    return x

             self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        else:
            raise NotImplementedError(f"local_module {local_module} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module] or None):
        super(OpSequential, self).__init__()
        op_list = op_list or []
        self.op_list = nn.ModuleList(op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(ResBlock, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or in_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

        if in_channels == out_channels and stride == 1:
            self.skip_connect = nn.Identity()
        else:
            self.skip_connect = ConvLayer(
                in_channels,
                out_channels,
                1,
                stride,
                use_bias=False,
                norm=norm[0],
                act_func=None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.skip_connect(x)
        return out + residual


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


from mfplnet.models.registry import BACKBONES


__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]


class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        use_dcnv4_in_stages=None,
        dcnv4_kwargs=None,
    ) -> None:
        super().__init__()

        self.width_list = []
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,

            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )

                if stride == 1:
                    shortcut = IdentityLayer()
                else:
                    shortcut = ConvLayer(
                        in_channels=in_channels,
                        out_channels=w,
                        kernel_size=1,
                        stride=stride,
                        norm=norm,
                        act_func=None,
                    )
                block = ResidualBlock(block, shortcut)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )

            shortcut_first_block = ConvLayer(
                in_channels=in_channels,
                out_channels=w,
                kernel_size=1,
                stride=2,
                norm=norm,
                act_func=None,
            )
            stage.append(ResidualBlock(block, shortcut_first_block))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,

    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block




    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: Optional[list[str]] = None,
        expand_list: Optional[list[float]] = None,
        fewer_norm_list: Optional[list[bool]] = None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()
        block_list = ["res", "fmb", "fmb", "mb", "att"] if block_list is None else block_list
        expand_list = [1, 4, 4, 4, 6] if expand_list is None else expand_list
        fewer_norm_list = [False, False, False, True, True] if fewer_norm_list is None else fewer_norm_list

        self.width_list = []
        self.stages = []
        stage0 = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )

            shortcut_first_block = ConvLayer(
                in_channels=in_channels,
                out_channels=w,
                kernel_size=1,
                stride=2,
                norm=norm,
                act_func=None,
            )
            stage.append(ResidualBlock(block, shortcut_first_block))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def _get_backbone_fn(model_name: str):
    backbone_fns = {
        'EfficientViT_MIT_B0': efficientvit_backbone_b0,
        'EfficientViT_MIT_B1': efficientvit_backbone_b1,
        'EfficientViT_MIT_B2': efficientvit_backbone_b2,
        'EfficientViT_MIT_B3': efficientvit_backbone_b3,
        'EfficientViT_MIT_L0': efficientvit_backbone_l0,
        'EfficientViT_MIT_L1': efficientvit_backbone_l1,
        'EfficientViT_MIT_L2': efficientvit_backbone_l2,
        'EfficientViT_MIT_L3': efficientvit_backbone_l3,
    }
    fn = backbone_fns.get(model_name)
    if fn is None:
        raise ValueError(f"Unsupported MIT EfficientViT model name: {model_name}. "
                         f"Available options are: {list(backbone_fns.keys())}")
    return fn


@BACKBONES.register_module
class MIT_EfficientViTWrapper(nn.Module):
    def __init__(self,
                 model_name='EfficientViT_MIT_B0',
                 pretrained=False,
                 pretrained_path=None,
                 out_indices=(1, 2, 3),
                 frozen_stages=0,
                 norm_eval=False,
                 cfg=None,
                 ):
        super(MIT_EfficientViTWrapper, self).__init__()
        self.cfg = cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        backbone_fn = _get_backbone_fn(model_name)

        self.model = backbone_fn()

        if self.frozen_stages >= 0:
            if hasattr(self.model, 'input_stem'):
                self.model.input_stem.eval()
                for param in self.model.input_stem.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 1:
            if hasattr(self.model, 'stages'):
                for i in range(min(self.frozen_stages - 1, len(self.model.stages))):
                    stage = self.model.stages[i]
                    stage.eval()
                    for param in stage.parameters():
                        param.requires_grad = False

        if pretrained_path:
            print(f"Loading pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')

            if 'state_dict' in state_dict:
                 state_dict = state_dict['state_dict']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully (strict=False).")
        elif pretrained:
             print(f"Warning: 'pretrained={pretrained}' but no 'pretrained_path' provided. Initializing randomly.")

    def forward(self, x):
        features_dict = self.model(x)

        outs = []
        for i in self.out_indices:
            stage_key = f'stage{i}'
            if stage_key not in features_dict:
                raise KeyError(f"Stage '{stage_key}' not found in backbone output. Available keys: {list(features_dict.keys())}")
            outs.append(features_dict[stage_key])

        return outs

    def train(self, mode=True):
        super(MIT_EfficientViTWrapper, self).train(mode)

        if self.frozen_stages >= 0:
            if hasattr(self.model, 'input_stem'):
                self.model.input_stem.eval()
                for param in self.model.input_stem.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 1:
            if hasattr(self.model, 'stages'):
                for i in range(min(self.frozen_stages - 1, len(self.model.stages))):
                    stage = self.model.stages[i]
                    stage.eval()
                    for param in stage.parameters():
                        param.requires_grad = False

        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()