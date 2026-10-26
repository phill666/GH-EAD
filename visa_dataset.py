from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


DEFAULT_VISA_SPLIT_CSV = '1cls.csv'


@dataclass(frozen=True)
class VisaRecord:
    object: str
    split: str
    label: str
    image: str
    mask: str | None

    @property
    def is_normal(self) -> bool:
        return self.label == 'normal'

    @property
    def is_anomaly(self) -> bool:
        return not self.is_normal


def _normalize_mask_value(mask_value: str | None) -> str | None:
    if mask_value is None:
        return None
    normalized = str(mask_value).strip()
    return normalized or None


def resolve_visa_layout(
    dataset_root: str | Path,
    category: str,
    split_csv_file: str = DEFAULT_VISA_SPLIT_CSV,
) -> dict[str, str]:
    dataset_root = Path(dataset_root)
    candidate_layouts = [
        (dataset_root, dataset_root / category),
        (dataset_root.parent, dataset_root),
    ]

    checked = []
    for root_candidate, category_candidate in candidate_layouts:
        split_csv_path = root_candidate / 'split_csv' / split_csv_file
        images_dir = category_candidate / 'Data' / 'Images'
        checked.append(
            {
                'dataset_root': str(root_candidate),
                'category_root': str(category_candidate),
                'split_csv_path': str(split_csv_path),
            }
        )
        if split_csv_path.is_file() and images_dir.is_dir():
            return {
                'dataset_root': str(root_candidate.resolve()),
                'category_root': str(category_candidate.resolve()),
                'split_csv_path': str(split_csv_path.resolve()),
            }

    raise FileNotFoundError(
        'Could not resolve VisA dataset root/category root from '
        f'dataset_root={dataset_root} and category={category}. Checked: {checked}'
    )


def load_visa_records(
    dataset_root: str | Path,
    category: str,
    split: str | None = None,
    split_csv_file: str = DEFAULT_VISA_SPLIT_CSV,
) -> list[VisaRecord]:
    layout = resolve_visa_layout(dataset_root, category, split_csv_file=split_csv_file)
    split_csv_path = Path(layout['split_csv_path'])

    records: list[VisaRecord] = []
    with split_csv_path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get('object') != category:
                continue
            current_split = str(row.get('split', '')).strip()
            if split is not None and current_split != split:
                continue
            records.append(
                VisaRecord(
                    object=str(row.get('object', '')).strip(),
                    split=current_split,
                    label=str(row.get('label', '')).strip(),
                    image=str(row.get('image', '')).strip(),
                    mask=_normalize_mask_value(row.get('mask')),
                )
            )
    return records


def infer_visa_defect_class(image_path: str | Path) -> str:
    defect_class = Path(image_path).parent.name.lower()
    return 'good' if defect_class == 'normal' else defect_class


def resolve_visa_mask_path(image_path: str | Path) -> Path | None:
    image_path = Path(image_path)
    if image_path.parent.name.lower() == 'normal':
        return None

    mask_path = image_path.parents[2] / 'Masks' / image_path.parent.name / f'{image_path.stem}.png'
    if mask_path.is_file():
        return mask_path

    raise FileNotFoundError(f'Could not find VisA mask for image: {image_path}')


class VisaImageDataset(Dataset):
    def __init__(self, dataset_root, records, transform=None, return_path=False):
        self.dataset_root = Path(dataset_root)
        self.records = list(records)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = (self.dataset_root / record.image).resolve()
        with Image.open(image_path) as image:
            sample = image.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.return_path:
            target = 0 if record.is_normal else 1
            return sample, target, str(image_path)
        return sample
