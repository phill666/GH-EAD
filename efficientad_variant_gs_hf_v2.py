from __future__ import annotations

import torch
from torch import nn

from common import get_autoencoder, get_pdn_medium, get_pdn_small
from modules.gs_halfconv_v2 import GSHalfConvV2
from modules.hf_igate_v2 import HFIGateEvidenceFusionV2


BASELINE_VARIANT = 'baseline'
GS_STUDENT_ONLY_VARIANT = 'gs_student_only'
GS_AE_ONLY_VARIANT = 'gs_ae_only'
GS_AE_HF_VARIANT = 'gs_ae_hf'
GS_STUDENT_CONV1_ONLY_VARIANT = 'gs_student_conv1_only'
GS_STUDENT_CONV1_HF_VARIANT = 'gs_student_conv1_hf'
GS_STUDENT_CONV2_ONLY_VARIANT = 'gs_student_conv2_only'
GS_HF_FULL_VARIANT = 'gs_hf_full'

# Legacy variants kept only so old runs/configs can still be reproduced.
LEGACY_GS_ONLY_VARIANT = 'gs_only'
LEGACY_HF_ONLY_VARIANT = 'hf_only'
LEGACY_GS_HF_FULL_VARIANT = 'variant_gs_hf_v2'
LEGACY_RESIDUAL_VARIANT = 'variant_gs_hf_v2_residual'

PUBLIC_VARIANTS = {
    BASELINE_VARIANT,
    GS_STUDENT_ONLY_VARIANT,
    GS_AE_ONLY_VARIANT,
    GS_AE_HF_VARIANT,
    GS_STUDENT_CONV1_ONLY_VARIANT,
    GS_STUDENT_CONV1_HF_VARIANT,
    GS_STUDENT_CONV2_ONLY_VARIANT,
    GS_HF_FULL_VARIANT,
}
LEGACY_VARIANTS = {
    LEGACY_GS_ONLY_VARIANT,
    LEGACY_HF_ONLY_VARIANT,
    LEGACY_RESIDUAL_VARIANT,
}
SUPPORTED_VARIANTS = PUBLIC_VARIANTS | LEGACY_VARIANTS
VARIANT_ALIASES = {
    LEGACY_GS_HF_FULL_VARIANT: GS_HF_FULL_VARIANT,
}


def normalize_model_variant(variant: str | None) -> str:
    normalized = str(variant or BASELINE_VARIANT).strip().lower()
    normalized = VARIANT_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_VARIANTS:
        raise ValueError(
            'Unsupported model variant: '
            f'{variant}. Expected one of {sorted(PUBLIC_VARIANTS)}.'
        )
    return normalized


class ResidualEvidenceFusionV2(nn.Module):
    """Legacy residual fusion kept only for backward compatibility."""

    def __init__(self, hf_fuser: nn.Module, beta: float = 0.2):
        super().__init__()
        self.hf_fuser = hf_fuser
        self.beta = float(beta)

    def forward(self, map_st, map_ae):
        base_map = 0.5 * map_st + 0.5 * map_ae
        hf_map = self.hf_fuser(map_st, map_ae)
        final_map = (1.0 - self.beta) * base_map + self.beta * hf_map
        return torch.nan_to_num(final_map, nan=0.0, posinf=0.0, neginf=0.0)


def _student_gs_conv_indices(model_size: str, variant: str) -> tuple[int, ...]:
    variant = normalize_model_variant(variant)

    if model_size == 'small':
        full_indices = (1, 2)
    elif model_size == 'medium':
        # Medium PDN has extra projection heads; these are the semantic
        # counterparts of the small-model mid-late GS insertion points.
        full_indices = (2, 3)
    else:
        raise ValueError(f'Unsupported model_size: {model_size}')

    if variant in {
        GS_STUDENT_ONLY_VARIANT,
        GS_HF_FULL_VARIANT,
        LEGACY_GS_ONLY_VARIANT,
        LEGACY_RESIDUAL_VARIANT,
    }:
        return full_indices
    if variant == GS_STUDENT_CONV1_ONLY_VARIANT:
        return (full_indices[0],)
    if variant == GS_STUDENT_CONV1_HF_VARIANT:
        return (full_indices[0],)
    if variant == GS_STUDENT_CONV2_ONLY_VARIANT:
        return (full_indices[1],)
    return ()


def variant_uses_student_gs(model_size: str, variant: str) -> bool:
    return len(_student_gs_conv_indices(model_size, variant)) > 0


def variant_uses_ae_gs(variant: str) -> bool:
    variant = normalize_model_variant(variant)
    return variant in {
        GS_AE_ONLY_VARIANT,
        GS_AE_HF_VARIANT,
        GS_HF_FULL_VARIANT,
        LEGACY_GS_ONLY_VARIANT,
        LEGACY_RESIDUAL_VARIANT,
    }


def variant_uses_hf(variant: str) -> bool:
    variant = normalize_model_variant(variant)
    return variant in {
        GS_AE_HF_VARIANT,
        GS_STUDENT_CONV1_HF_VARIANT,
        GS_HF_FULL_VARIANT,
        LEGACY_HF_ONLY_VARIANT,
        LEGACY_RESIDUAL_VARIANT,
    }


def _insert_gs_after_blocks(
    backbone: nn.Sequential,
    target_conv_indices: tuple[int, ...],
) -> nn.Sequential:
    target_set = set(target_conv_indices)
    if not target_set:
        return backbone

    layers = []
    conv_index = 0
    pending_channels = None

    for module in backbone:
        layers.append(module)
        if isinstance(module, nn.Conv2d):
            if conv_index in target_set:
                pending_channels = module.out_channels
            else:
                pending_channels = None
            conv_index += 1
            continue

        if pending_channels is not None and isinstance(module, nn.ReLU):
            layers.append(GSHalfConvV2(pending_channels))
            pending_channels = None

    if pending_channels is not None:
        layers.append(GSHalfConvV2(pending_channels))

    return nn.Sequential(*layers)


def build_student_model(model_size: str, out_channels: int, padding: bool, variant: str):
    variant = normalize_model_variant(variant)
    if model_size == 'small':
        backbone = get_pdn_small(2 * out_channels, padding=padding)
    elif model_size == 'medium':
        backbone = get_pdn_medium(2 * out_channels, padding=padding)
    else:
        raise ValueError(f'Unsupported model_size: {model_size}')

    if variant_uses_student_gs(model_size, variant):
        return _insert_gs_after_blocks(
            backbone,
            target_conv_indices=_student_gs_conv_indices(model_size, variant),
        )
    return backbone


def build_autoencoder_model(out_channels: int, variant: str):
    variant = normalize_model_variant(variant)
    autoencoder = get_autoencoder(out_channels)
    if not variant_uses_ae_gs(variant):
        return autoencoder

    # Only lightly touch the encoder's latter two 64-channel blocks.
    return _insert_gs_after_blocks(autoencoder, target_conv_indices=(3, 4))


def build_evidence_fuser(variant: str, hf_fusion_beta: float = 0.2):
    variant = normalize_model_variant(variant)
    if variant_uses_hf(variant):
        hf_fuser = HFIGateEvidenceFusionV2(hf_scale_init=0.1)
        if variant == LEGACY_RESIDUAL_VARIANT:
            return ResidualEvidenceFusionV2(hf_fuser, beta=hf_fusion_beta)
        return hf_fuser
    return None
