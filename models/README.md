# Model definitions

Executable definitions retain original import names for checkpoint compatibility. models/efficientad/model.py and models/gshconv and models/hicgate re-export canonical functions/classes without duplicating complete alternative models.

EfficientAD: RGB 256 -> teacher [B,384,56,56], student [B,768,56,56], AE feature output [B,384,56,56]. The teacher is frozen. Teacher/former-student and AE/latter-student squared discrepancies are averaged across 384 channels.

GSHConv preserves [B,C,H,W] feature shape. Formal default: groups 2, per-group partial convolution on floor(group_channels/2), shuffle ON, depthwise 3x3 + pointwise 1x1 + GELU, SE-style gate ON, BatchNorm and learned residual scale initialized to 0.1. Full inserts two blocks after student Conv indices 1/2 and two after AE encoder Conv indices 3/4 (zero-based, following ReLU). Do not use a sensitivity-test winner or the old manuscript g3 description to redefine defaults.

HICGate: two normalized anomaly responses [B,1,56,56] -> fused response of the same shape. Fixed kernel [0,-1,0;-1,4,-1;0,-1,0], fixed scale 0.1 per branch, HF nonnegative clamp, gate=sigmoid((U_l-U_g)/(abs(U_l-U_g)+U_l+U_g+1e-6)), output=gate*U_l+(1-gate)*U_g. Optional bilinear alignment and finite cleanup are present. Zero learnable parameters. No MLP,CBR,separate Spatial/Mix/Coordinate Attention or multichannel feature fusion.

Pretrained models/teacher_small.pth is required but not redistributed. Obtain it separately from the [public, unofficial EfficientAD implementation](https://github.com/nelson1425/EfficientAD), which provides pretrained teacher weights and pretraining instructions. Random-teacher fallback is forbidden. environment/teacher_reference.json records the observed local weight fingerprint; it is not proof that every upstream release has identical bytes.

cfrp_dataset.py is retained for import compatibility only; its dataset is outside the three supported public commands and no data or split files for it are supplied.

The retained config field `model.hf_fusion_beta=0.2` controls only a legacy residual wrapper. It is inactive for `gs_hf_full` and is not the HF scale of HICGate v2, whose two buffers are 0.1.
