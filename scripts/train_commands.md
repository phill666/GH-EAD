# Training commands — prepared only

Run at the package root in the existing compatible environment after separately acquiring datasets/pretrained teacher. Relative data/output paths are examples. Every target must be new; no overwrite, automatic retry or parallel queue. Protocol: seed42,70000 steps, RGB 256,batch 1,Adam lr 1e-4 / wd 1e-5,workers 4. Defaults do not follow sensitivity-test winners.

## mvtec_ad2

PowerShell:
```powershell
python -B scripts/train/train.py --dataset mvtec_ad2 --category can --data-root "data/mvtec_ad2" --output-root "outputs/train/mvtec_ad2" --variant gs_hf_full
```

cmd:
```bat
python -B scripts/train/train.py --dataset mvtec_ad2 --category can --data-root "data/mvtec_ad2" --output-root "outputs/train/mvtec_ad2" --variant gs_hf_full
```

All categories, sequentially:
```powershell
python -B scripts/train/train.py --dataset mvtec_ad2 --category all --data-root "data/mvtec_ad2" --output-root "outputs/train/mvtec_ad2" --variant gs_hf_full
```
## visa

PowerShell:
```powershell
python -B scripts/train/train.py --dataset visa --category candle --data-root "data/visa" --output-root "outputs/train/visa" --variant gs_hf_full
```

cmd:
```bat
python -B scripts/train/train.py --dataset visa --category candle --data-root "data/visa" --output-root "outputs/train/visa" --variant gs_hf_full
```

All categories, sequentially:
```powershell
python -B scripts/train/train.py --dataset visa --category all --data-root "data/visa" --output-root "outputs/train/visa" --variant gs_hf_full
```
## btad

PowerShell:
```powershell
python -B scripts/train/train.py --dataset btad --category 01 --data-root "data/btad" --output-root "outputs/train/btad" --variant gs_hf_full
```

cmd:
```bat
python -B scripts/train/train.py --dataset btad --category 01 --data-root "data/btad" --output-root "outputs/train/btad" --variant gs_hf_full
```

All categories, sequentially:
```powershell
python -B scripts/train/train.py --dataset btad --category all --data-root "data/btad" --output-root "outputs/train/btad" --variant gs_hf_full
```
