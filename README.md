# GH-EAD: Local-Global Collaborative Enhancement for Efficient Industrial Visual Anomaly Detection

## 1. Overview

This package contains the audited EfficientAD-based GH-EAD implementation, dataset loaders, source-derived configurations, fixed-last training/testing entry points and the shared revision evaluator. It uses GSHConv in the student/autoencoder feature towers and HICGate v2 to fuse **single-channel anomaly responses**, not multichannel feature tensors.

Datasets, masks and checkpoints are not included. This is a local prepared source tree. Remote publication status must be verified separately; a target URL is not evidence of an uploaded repository. `SOURCE_PROVENANCE.json` records source fingerprints; `NOTICE.md` and `LICENSE` retain attribution. Original project files and paper results are unchanged.

## 2. Environment

The observed revision environment is Python 3.10.20, PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, CUDA build 11.8 and cuDNN 8700. See `environment/environment.txt` for the full observed versions and `environment/requirements_minimal.txt` for actual runtime dependencies. These files describe the current environment; they do not instruct an upgrade or an automatic installation. A different environment must be disclosed rather than silently described as identical.

The manuscript's historical CUDA version and this observed revision environment are separate records. CUDA/cuDNN build versions are not a GPU-driver version. Model training requires a compatible GPU; help, source audits and category aggregation do not.

## 3. Dataset preparation

Follow `datasets/README.md`. Obtain public data from their providers; respect licensing. Set `--data-root` to the prepared directory. `datasets/split_manifest_seed42.json` lists relative filenames for the observed local splits, without distributing image data.

The configured categories are:

- MVTec AD 2: can, sheet_metal, vial, wallplugs, with official normal validation and public test.
- VisA: all 12 categories. The official training pool of 8659 is split into 7789 optimization samples and 870 internal normal-validation samples; test has 2162 samples.
- BTAD: 01, 02, 03. The observed local normal pool has 1799 samples, partitioned into 1619 optimization and 180 validation samples; test has 741. Category 02 has 399 local training samples.

Do not alter test sets or silently add data to reconcile differences with another archive. Verify filename sets and internal-split ordering before claiming the same protocol.

## 4. Directory structure

```text
configs/{mvtec_ad2,visa,btad}/ghead.yaml
configs/{mvtec_ad2,visa,btad}/sanitized_source_protocol.json
configs/CHECKPOINT_POLICY.md
models/{efficientad,gshconv,hicgate}/
modules/                    # canonical GSHConv and HICGate definitions
datasets/                   # preparation guide and relative split manifest
scripts/train/train.py
scripts/test/test.py
scripts/evaluate/aggregate.py
scripts/train_commands.md
scripts/test_commands.md
metrics/                    # metric guide and re-export
environment/                # observed versions and teacher fingerprint
results_example/            # schema only; no invented experimental values
```

Canonical root-level implementation files retain their import names for checkpoint compatibility. Model re-exports avoid duplicating alternative architectures.

## 5. Training

Acquire the pretrained teacher separately as `models/teacher_small.pth`; see `models/README.md` and `environment/teacher_reference.json`. Random-teacher fallback is forbidden by the public launcher. The observed teacher fingerprint is a local reference; upstream byte identity has not been independently certified.

Run from this package root in PowerShell or cmd:

```powershell
python -B scripts/train/train.py --dataset mvtec_ad2 --category can --data-root "data/mvtec_ad2" --output-root "outputs/train/mvtec_ad2" --variant gs_hf_full
```

`--category all` runs the configured categories sequentially. Commands for all three datasets, in both Windows shells, are in `scripts/train_commands.md`. Training is never started by documentation preparation. Every target run directory must be new; existing directories are rejected.

Fixed settings are RGB 256 x 256, seed 42, batch size 1, 70000 steps, Adam with initial learning rate 0.0001 and weight decay 0.00001, and 4 workers. GSHConv uses groups 2, half ratio 0.50, shuffle ON, SE gate ON and residual-scale initialization 0.1. HICGate has fixed HF scales 0.1 and zero learnable parameters. Later sensitivity results do not redefine these defaults.

## 6. Evaluation

Use the predefined 70000-step **last** bundle. An explicit run directory or template is required; no best/latest fallback is exposed.

```powershell
python -B scripts/test/test.py --dataset mvtec_ad2 --category can --data-root "data/mvtec_ad2" --output-root "outputs/eval/mvtec_ad2/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/mvtec_ad2/gs_hf_full/can/seed42"
```

After every category has been evaluated:

```powershell
python -B scripts/evaluate/aggregate.py --dataset mvtec_ad2 --input-root "outputs/eval/mvtec_ad2/gs_hf_full" --output-dir "outputs/tables/mvtec_ad2/gs_hf_full"
```

The evaluator computes Image AUROC, Pixel AUROC and AU-PRO on the complete test split. Its implementation is retained from the revision project; no tuning of thresholds, FPR cap or scores is introduced. All-category and other-dataset commands are in `scripts/test_commands.md`.

## 7. Reproducing main results

Three source-derived Full configurations are supplied. Their sanitized historical source protocols are retained separately for transparency. Public entry points disable intermediate test monitoring and top-k selection and use separate fixed-last evaluation; see `configs/CHECKPOINT_POLICY.md`.

This explicit checkpoint/evaluation policy differs from historical source configurations. Removing intermediate evaluation may affect RNG consumption. Thus fresh fixed-last runs are a disclosed cohort, and exact bitwise replay or equality with every historical paper aggregate is not claimed. No historical table values are changed. The package contains no trained GH-EAD checkpoint; future users can generate a final bundle with the training commands.

## 8. Reproducing ablation results

The launchers support the four module configurations `baseline`, `gs_only`, `hf_only` and `gs_hf_full`, using the canonical factory. Choose a separate output root for each configuration and report all three metrics with category-equal macros.

This minimal package does not include an automatic recreation of every completed internal GSHConv sensitivity run or every HICGate inference-mode ablation. Those experiments require their preregistered variant configurations and calibration-preserving runners. Do not redefine Full defaults or use test results to select a variant. Adding these archived runners is a separate extension, not a prerequisite for the scoped main-model package.

## 9. Random seed

The frozen configs use seed 42 and the original seed/split logic. `deterministic=false` is retained from resolved source configs, so a fixed seed is not a guarantee of bitwise identity across hardware or kernels. Workers, augmentation, optimizer, model initialization and checkpoint policy must be reported. The package does not start multiseed training.

## 10. Metrics

See `metrics/README.md`. Image scores are final anomaly-map maxima. Pixel metrics include normal and anomalous test images in full native image domains. Normal masks are zero; official anomalous masks are nearest-neighbor aligned when required. The fused 56 x 56 map is padded by four pixels per edge and bilinearly resized to native image dimensions. AU-PRO uses 8-connected components and FPR cap 0.3. Dataset macros give each category equal weight.

Teacher statistics use optimization training; map quantiles use normal validation. No test-dependent recalibration, threshold selection or pixel filtering is allowed.

## 11. Expected outputs

Training produces the resolved configuration, command, logs, and `checkpoints/last/` with teacher, student, autoencoder, fusion module where applicable, metadata and final-bundle hashes. Evaluation saves the command, log, checkpoint hashes and metrics, plus prediction outputs defined by the shared evaluator. Aggregation creates per-category/macro CSVs and input metric-file hashes. `results_example/metrics_schema.json` is a schema, not an experimental result.

Only static parsing, command-line help, configuration loading, a synthetic CPU shape/formula check and CPU aggregation checks were performed during preparation. End-to-end GPU training/evaluation has not been executed to certify this release candidate.

## 12. Troubleshooting

Missing data or pretrained teacher: provide the required external resources; never substitute random weights. Existing output: keep it and select a fresh output root. Incomplete or mixed-category metrics: resolve the missing evaluation without computing a partial macro. Checkpoint mismatch: stop and verify run identity rather than selecting another checkpoint by score.

On Windows DataLoader, CUDA, WHEA or system failures, preserve logs and outputs and investigate system stability. Do not silently change workers, batch size, seed or protocol to label a retry equivalent. If model imports fail, run from the package root with the documented existing environment. Do not upgrade dependencies as an automatic repair.

## Paper and code availability

Manuscript under review at Image and Vision Computing. No acceptance, DOI, volume, issue or publication year is claimed. The requested repository target is https://github.com/phill666/GH-EAD; availability and visibility must be verified after authenticated private publication. Do not cite the target as a published public repository before verification. Weights and datasets are not bundled, and no unverified download link is offered.

## Installation

Use the recorded environment for the closest match. The CUDA11.8 wheel build is recorded explicitly. In a separate user-managed environment, resolve the torch/torchvision CUDA wheel source before installing requirements.txt. Existing environment must not be upgraded automatically. No dependencies were installed during release preparation.

## Ablations and efficiency

See docs/ABLATIONS.md and configs/ablation for structural sensitivities and seed handling. See docs/EFFICIENCY.md for the actual archived counting/timing boundary. HIC calibration-preserving internal ablation and a portable complete benchmark runner remain documented extensions, not falsely advertised as fully verified entry points. Preparation verifies syntax/imports/configs/help and a small CPU module check only, not new training or full dataset evaluation.

## Acknowledgments and licensing

The foundation is nelson1425/EfficientAD, a public unofficial implementation of EfficientAD. Retain its Apache-2.0 LICENSE and see NOTICE.md for upstream attribution and modification notices. No third-party dataset or pretrained-weight redistribution rights are implied. Obtain data and teacher weights separately and follow provider terms.

## Citation

Pending formal publication details, cite the manuscript title: GH-EAD: Local-Global Collaborative Enhancement for Efficient Industrial Visual Anomaly Detection (manuscript under review). A formal BibTeX record and publication metadata will be added when available; no DOI or author metadata is invented by this package.
