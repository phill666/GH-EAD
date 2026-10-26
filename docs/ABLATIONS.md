# Ablations and verification limits

Module variants baseline / gs_only / hf_only / gs_hf_full use the main train/test commands with independent output roots. GSH structural changes require new training; inference mode switches must not be described as retrained models.

Example (not executed during release preparation):
```
python -B scripts/ablation/train_sensitivity.py --groups 1 --dataset mvtec_ad2 --category can --data-root data/mvtec_ad2 --output-root outputs/g1 --variant gs_hf_full --seed 42
```
Use --half-ratio .25/.50/.75, --no-shuffle, --no-se or --groups 1/2/3/4. Config recipes are in configs/ablation. The exact archived ConfigurableGSHalfConvV2 extends unequal groups by an index permutation when shuffle is ON; therefore do not claim g3 always disables shuffle. This extension is distinct from the canonical model's divisibility fallback and never changes the default g2 behavior.

Three-seed training accepts --seed 42, 2025 or 2026 with independent output roots. Default seed42 remains unchanged. The stock test launcher is seed42-specific; for other seeds invoke tools/evaluate_model.py with the run's resolved_config.json and explicit --run-dir / --checkpoint-kind last / --split / --save-dir. Do not accidentally evaluate another seed with the seed42 config. Use the shared category aggregation per seed and sample std ddof=1 across complete seed macros.

HIC internal modes are Base/HF-only/Gate-only/Full on the same Full checkpoint with exactly replayed training/normal-validation calibration. This release does not expose a fully portable runner for that calibration-preserving archived protocol; it must not claim end-to-end reproduction of that sensitivity table. Main default HIC code and formulas are included. No historical result is fabricated and no experiment is launched by documentation.
