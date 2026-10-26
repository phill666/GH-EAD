# Testing commands — prepared only

Explicit fixed run paths/templates are mandatory; no latest or best fallback. Require type=last,step 70000. Full original test split is used, with Image AUROC,Pixel AUROC,AU-PRO. Select a fresh evaluation root for any repeat. No subset flags,threshold/normalization overrides or test-driven configuration choices are exposed.

## mvtec_ad2

Single category (PowerShell):
```powershell
python -B scripts/test/test.py --dataset mvtec_ad2 --category can --data-root "data/mvtec_ad2" --output-root "outputs/eval/mvtec_ad2/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/mvtec_ad2/gs_hf_full/can/seed42"
```

Single category (cmd):
```bat
python -B scripts/test/test.py --dataset mvtec_ad2 --category can --data-root "data/mvtec_ad2" --output-root "outputs/eval/mvtec_ad2/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/mvtec_ad2/gs_hf_full/can/seed42"
```

All categories (same one-line command for either shell):
```powershell
python -B scripts/test/test.py --dataset mvtec_ad2 --category all --data-root "data/mvtec_ad2" --output-root "outputs/eval/mvtec_ad2/gs_hf_full" --variant gs_hf_full --run-root "outputs/train/mvtec_ad2"
```

Per-category and macro tables after all categories finish, CPU only:
```powershell
python -B scripts/evaluate/aggregate.py --dataset mvtec_ad2 --input-root "outputs/eval/mvtec_ad2/gs_hf_full" --output-dir "outputs/tables/mvtec_ad2/gs_hf_full"
```
## visa

Single category (PowerShell):
```powershell
python -B scripts/test/test.py --dataset visa --category candle --data-root "data/visa" --output-root "outputs/eval/visa/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/visa/gs_hf_full/candle/seed42"
```

Single category (cmd):
```bat
python -B scripts/test/test.py --dataset visa --category candle --data-root "data/visa" --output-root "outputs/eval/visa/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/visa/gs_hf_full/candle/seed42"
```

All categories (same one-line command for either shell):
```powershell
python -B scripts/test/test.py --dataset visa --category all --data-root "data/visa" --output-root "outputs/eval/visa/gs_hf_full" --variant gs_hf_full --run-root "outputs/train/visa"
```

Per-category and macro tables after all categories finish, CPU only:
```powershell
python -B scripts/evaluate/aggregate.py --dataset visa --input-root "outputs/eval/visa/gs_hf_full" --output-dir "outputs/tables/visa/gs_hf_full"
```
## btad

Single category (PowerShell):
```powershell
python -B scripts/test/test.py --dataset btad --category 01 --data-root "data/btad" --output-root "outputs/eval/btad/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/btad/gs_hf_full/01/seed42"
```

Single category (cmd):
```bat
python -B scripts/test/test.py --dataset btad --category 01 --data-root "data/btad" --output-root "outputs/eval/btad/gs_hf_full" --variant gs_hf_full --run-dir "outputs/train/btad/gs_hf_full/01/seed42"
```

All categories (same one-line command for either shell):
```powershell
python -B scripts/test/test.py --dataset btad --category all --data-root "data/btad" --output-root "outputs/eval/btad/gs_hf_full" --variant gs_hf_full --run-root "outputs/train/btad"
```

Per-category and macro tables after all categories finish, CPU only:
```powershell
python -B scripts/evaluate/aggregate.py --dataset btad --input-root "outputs/eval/btad/gs_hf_full" --output-dir "outputs/tables/btad/gs_hf_full"
```
