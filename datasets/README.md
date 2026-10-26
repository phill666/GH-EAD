# Dataset preparation

No images, masks or dataset archives are distributed. Acquire them from the authors and respect their terms. The relative filename manifest verifies the original local protocol rather than selecting a new split.

## MVTec AD 2

Use [MVTec's dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2). This package uses only can,sheet_metal,vial,wallplugs. It is not the full eight-category/private-server protocol. DATA_ROOT contains category directories; an extra nested category directory from local extraction is also supported by the resolver.

```text
data/mvtec_ad2/<category>/train/good/*
data/mvtec_ad2/<category>/validation/good/*
data/mvtec_ad2/<category>/test_public/good/*
data/mvtec_ad2/<category>/test_public/<defect>/*
data/mvtec_ad2/<category>/test_public/ground_truth/<defect>/*
```

Keep official normal validation separate. Local totals: train 1133, validation 139, public test 566 (normal 191 / anomaly 375). Do not substitute private test data or invent masks.

## VisA

Obtain original data and split_csv/1cls.csv from [the authors' repository](https://github.com/amazon-science/spot-diff). This loader reads original CSV records directly; no reorganized/random test split is introduced.

```text
data/visa/split_csv/1cls.csv
data/visa/<category>/Data/Images/Normal/*
data/visa/<category>/Data/Images/Anomaly/*
data/visa/<category>/Data/Masks/Anomaly/*
```

All 12 categories are configured. Official one-class train pool 8659 is partitioned into optimization 7789 and internal normal validation 870 for seed42; 870 is not an additional mutually exclusive pool to add to 8659. Test 2162 remains separate. Internal splitting uses original loader order,int(0.9*N),and a per-category torch.Generator seed42. Normal GT is zero; anomaly masks use official CSV records.

## BTAD

See [the authors' VT-ADL paper](https://arxiv.org/abs/2104.10036) and [AViReS laboratory site](https://avires.dimi.uniud.it/) for dataset access. A direct archive URL was not verified; no automatic download is supplied.

```text
data/btad/<01|02|03>/train/ok/*
data/btad/<01|02|03>/test/ok/*
data/btad/<01|02|03>/test/ko/*
data/btad/<01|02|03>/ground_truth/ko/*
```

Supported normal aliases and mask naming follow the original resolver. Observed local train pool 400 + 399 + 1000 = 1799; seed42 optimization/validation 1619/180; test 741. Category 02 retains 399 images. If a fresh official archive has additional training images, disclose the difference and compare with split_manifest_seed42.json rather than silently completing the local pool. Existing empty-GT anomaly-label test samples are retained, not filtered to improve scores.

## Split verification

split_manifest_seed42.json lists relative train/validation/test filenames for each configured category. Verify complete sets and loader ordering before reporting a reproduction. Different contents/order/seeds can alter the internal partition and must be disclosed. No public data binary or private dataset is included.
