import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import List

from ..ops.pointnet2.pointnet2_batch import pointnet2_utils


class DynamicSampling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        complexity_handler=None,
        activate: str = "gumbel"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.router = DynamicRouter(in_channels, out_channels, activate=activate)
        self.complexity_handler = complexity_handler
        self.running_gate = None

    def get_complexity(self):
        if self.complexity_handler is None:
            return {"static": None, "dynamic": None, "real": None}

        gate = self.running_gate
        B, Ng, Np = gate.shape

        num_points = gate.view(B, Ng, -1).float().sum(-1)
        num_inputs = num_points.new_full([B, Ng], Np, dtype=torch.float)
        num_real_points = (gate != 0).view(B, Ng, -1).float().sum(-1)

        dge_cp = self.router.complexity(num_inputs, num_points)
        dynamic_cp = self.complexity_handler(num_inputs, num_points) + dge_cp

        dge_real_cp = self.router.complexity(num_inputs, num_real_points)
        static_cp = self.complexity_handler(num_inputs, num_inputs) + dge_real_cp
        real_cp = self.complexity_handler(num_inputs, num_real_points) + dge_real_cp
        return {"static": static_cp, "dynamic": dynamic_cp, "real": real_cp}

    def routing(self, x):
        B = x.shape[0]
        gate = self.router(x)
        self.running_gate = gate

        if not self.training:
            indices = [gate[:, i].nonzero().contiguous() for i in range(self.out_channels)]
            return indices, [None for i in range(self.out_channels)]
        else:
            indices = [
                x.new_ones(B, gate.shape[-1]).nonzero().contiguous()
                for _ in range(self.out_channels)
            ]
            return indices, gate.split(1, dim=1)

    def compress(self, x, indices):
        if self.training:
            return x.transpose(1, 2).flatten(0, 1)
        x = pointnet2_utils.sparse_indexing_get(
            x.contiguous(), indices
        )
        return x

    def decompress(self, x, res, indices, gate=None):
        B, C, Np = res.shape

        if self.training:
            gate = gate.reshape(B, 1, Np)
            x = x.reshape(B, Np, -1).transpose(1, 2)
            return x * gate
        else:
            pointnet2_utils.sparse_indexing_replace(
                x, indices, res
            )
            return res


class DynamicRouter(nn.Module):

    def __init__(self, num_channels, num_outputs, init_gate=0.95, activate="gumbel"):
        super(DynamicRouter, self).__init__()
        self.num_channels = num_channels
        self.init_gate = init_gate
        self.activate = activate

        self.gate = nn.Conv1d(num_channels, num_outputs, kernel_size=1)
        self.init_parameters()

    def gate_activate(self, x):
        if self.activate == "rtanh":
            return F.relu(x.tanh())
        elif self.activate == "gumbel":
            if self.training:
                x = torch.stack([x, -x], dim=-1)
                return F.gumbel_softmax(x, dim=-1, hard=True)[..., 0]
            else:
                return x >= 0

    def init_parameters(self):
        bias_value = math.log(math.sqrt(
            self.init_gate / (1 - self.init_gate)))
        nn.init.constant_(self.gate.bias.data, bias_value)

    def complexity(self, num_inputs, num_points):
        comp = 0
        return comp

    def forward(self, x):
        x = self.gate(x)
        x = self.gate_activate(x)
        return x
