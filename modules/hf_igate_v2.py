from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HFIGateEvidenceFusionV2(nn.Module):
    """Evidence-only HF-IGate fusion used by variant_gs_hf_v2.

    This module intentionally stays out of the teacher/student/AE feature towers.
    It only fuses the normalized discrepancy maps `map_st` and `map_ae`.
    """

    def __init__(self, hf_scale_init: float = 0.1, eps: float = 1e-6):
        super().__init__()
        lap_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)
        self.register_buffer(
            'hf_scale_st',
            torch.tensor(float(hf_scale_init), dtype=torch.float32),
        )
        self.register_buffer(
            'hf_scale_ae',
            torch.tensor(float(hf_scale_init), dtype=torch.float32),
        )
        self.eps = float(eps)

    def _high_freq(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        weight = self.lap_kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, weight, stride=1, padding=1, groups=channels)

    def forward(self, map_st: Tensor, map_ae: Tensor) -> Tensor:
        if map_st.shape[-2:] != map_ae.shape[-2:]:
            map_ae = F.interpolate(
                map_ae,
                size=map_st.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        map_st = torch.nan_to_num(map_st, nan=0.0, posinf=0.0, neginf=0.0)
        map_ae = torch.nan_to_num(map_ae, nan=0.0, posinf=0.0, neginf=0.0)

        map_st_hf = torch.clamp_min(
            map_st + self.hf_scale_st * self._high_freq(map_st),
            0.0,
        )
        map_ae_hf = torch.clamp_min(
            map_ae + self.hf_scale_ae * self._high_freq(map_ae),
            0.0,
        )

        interaction = torch.abs(map_st_hf - map_ae_hf)
        gate_logits = (map_st_hf - map_ae_hf) / (interaction + map_st_hf + map_ae_hf + self.eps)
        gate = torch.sigmoid(gate_logits)

        fused = gate * map_st_hf + (1.0 - gate) * map_ae_hf
        return torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
