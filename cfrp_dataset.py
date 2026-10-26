from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:
        pass


DEFAULT_CFRP_ROOT = 'data/unused_compatibility_dataset'
DEFAULT_SPLIT_PATH = Path('configs') / 'cfrp_split.json'
IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def _natural_numeric_key(filename):
    stem = Path(filename).stem
    if stem.isdigit():
        return (0, int(stem), filename.lower())
    parts = re.split(r'(\d+)', stem)
    key = [int(part) if part.isdigit() else part.lower() for part in parts]
    return (1, key, filename.lower())


def resolve_cfrp_dirs(cfrp_root):
    root = Path(cfrp_root)
    candidates = [
        (
            root / 'originalData' / 'originalData',
            root / 'annotatedData' / 'annotatedData',
        ),
        (root / 'originalData', root / 'annotatedData'),
        (root / 'images', root / 'masks'),
    ]

    checked = []
    for image_dir, mask_dir in candidates:
        checked.append((str(image_dir), str(mask_dir)))
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir.resolve(), mask_dir.resolve()

    raise FileNotFoundError(
        'Could not find CFRP image/mask directories. Checked: '
        + '; '.join(f'{image_dir} | {mask_dir}' for image_dir, mask_dir in checked)
    )


def scan_cfrp_filenames(cfrp_root):
    image_dir, mask_dir = resolve_cfrp_dirs(cfrp_root)
    image_names = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    mask_names = {
        path.name
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    filenames = sorted(image_names & mask_names, key=_natural_numeric_key)
    if not filenames:
        raise FileNotFoundError(
            f'No paired CFRP image/mask files found in {image_dir} and {mask_dir}.'
        )
    return filenames, image_dir, mask_dir


def count_cfrp_files(cfrp_root):
    image_dir, mask_dir = resolve_cfrp_dirs(cfrp_root)
    image_names = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    mask_names = {
        path.name
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    matched = sorted(image_names & mask_names, key=_natural_numeric_key)
    return image_names, mask_names, matched, image_dir, mask_dir


def split_cfrp_filenames(filenames):
    filenames = list(filenames)
    total = len(filenames)
    train_end = int(total * 0.7)
    val_end = int(total * 0.8)
    return {
        'train': filenames[:train_end],
        'val': filenames[train_end:val_end],
        'test': filenames[val_end:],
    }


def build_cfrp_split(cfrp_root, split_path=DEFAULT_SPLIT_PATH, write=True):
    filenames, image_dir, mask_dir = scan_cfrp_filenames(cfrp_root)
    split = split_cfrp_filenames(filenames)
    if write and split_path is not None:
        split_path = Path(split_path)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open('w', encoding='utf-8') as file:
            json.dump(split, file, indent=2)
    return split, image_dir, mask_dir


def load_or_create_cfrp_split(cfrp_root, split_path=DEFAULT_SPLIT_PATH):
    return build_cfrp_split(cfrp_root, split_path=split_path, write=True)


def _load_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def _load_binary_mask(path):
    with Image.open(path) as mask:
        mask = mask.convert('L')
    return mask.point(lambda value: 255 if value > 0 else 0, mode='L')


def binary_mask_sum(path):
    with Image.open(path) as mask:
        histogram = mask.convert('L').histogram()
    return int(sum(histogram[1:]))


def summarize_split(split_name, filenames, mask_dir):
    mask_sums = [binary_mask_sum(mask_dir / filename) for filename in filenames]
    label_1 = sum(1 for value in mask_sums if value > 0)
    label_0 = len(mask_sums) - label_1
    if mask_sums:
        minimum = min(mask_sums)
        maximum = max(mask_sums)
        mean = statistics.fmean(mask_sums)
    else:
        minimum = maximum = mean = 0.0
    return {
        'split': split_name,
        'count': len(filenames),
        'label_0': label_0,
        'label_1': label_1,
        'mask_sum_min': minimum,
        'mask_sum_max': maximum,
        'mask_sum_mean': mean,
        'mask_sums': mask_sums,
    }


class CfrpTrainDataset(Dataset):
    def __init__(self, cfrp_root, split='train', transform=None, split_path=DEFAULT_SPLIT_PATH):
        split_data, image_dir, _ = load_or_create_cfrp_split(cfrp_root, split_path)
        if split not in split_data:
            raise ValueError(f'Unsupported CFRP split: {split}')
        self.image_dir = image_dir
        self.filenames = list(split_data[split])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = _load_rgb(self.image_dir / self.filenames[index])
        if self.transform is not None:
            image = self.transform(image)
        return image


class CfrpEvalDataset(Dataset):
    def __init__(self, cfrp_root, split='test', transform=None, split_path=DEFAULT_SPLIT_PATH):
        split_data, image_dir, mask_dir = load_or_create_cfrp_split(cfrp_root, split_path)
        if split == 'all':
            filenames = split_data['train'] + split_data['val'] + split_data['test']
        elif split in split_data:
            filenames = split_data[split]
        else:
            raise ValueError(f'Unsupported CFRP split: {split}')
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = self.image_dir / filename
        mask_path = self.mask_dir / filename
        image = _load_rgb(image_path)
        mask = _load_binary_mask(mask_path)
        label = int(mask.getbbox() is not None)
        if self.transform is not None:
            image = self.transform(image)
        return image, mask, label, str(image_path), filename


def collate_cfrp_eval_batch(batch):
    if torch is None:
        raise RuntimeError('PyTorch is required to use CFRP DataLoader collation.')
    images, masks, labels, paths, filenames = zip(*batch)
    return list(images), list(masks), torch.tensor(labels, dtype=torch.long), list(paths), list(filenames)


def check_cfrp_dataset(cfrp_root, split_path=DEFAULT_SPLIT_PATH):
    image_names, mask_names, matched, image_dir, mask_dir = count_cfrp_files(cfrp_root)
    split = split_cfrp_filenames(matched)
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open('w', encoding='utf-8') as file:
        json.dump(split, file, indent=2)

    total = sum(len(items) for items in split.values())
    print(f'CFRP root: {Path(cfrp_root).resolve()}')
    print(f'Image dir: {image_dir}')
    print(f'Mask dir: {mask_dir}')
    print(f'Original image files: {len(image_names)}')
    print(f'Mask files: {len(mask_names)}')
    print(f'Valid matched samples: {len(matched)}')
    print(f'Total split samples: {total}')
    print(f"Train: {len(split['train'])}")
    print(f"Val: {len(split['val'])}")
    print(f"Test: {len(split['test'])}")
    print(f'Split file: {Path(split_path).resolve()}')
    print('')

    summaries = {
        name: summarize_split(name, filenames, mask_dir)
        for name, filenames in split.items()
    }
    for name in ('train', 'val', 'test'):
        summary = summaries[name]
        print(
            f"{name}: count={summary['count']} "
            f"label=0:{summary['label_0']} label=1:{summary['label_1']} "
            f"mask_sum min={summary['mask_sum_min']} "
            f"max={summary['mask_sum_max']} "
            f"mean={summary['mask_sum_mean']:.2f}"
        )
        print(f'{name} first 10 filename,label,mask_sum:')
        for filename, mask_sum in zip(split[name][:10], summary['mask_sums'][:10]):
            print(f'  {filename},label={int(mask_sum > 0)},mask_sum={mask_sum}')
        print('')

    train_summary = summaries['train']
    if train_summary['label_1'] > 0:
        print(
            'WARNING: The current chronological CFRP split has defect-labeled '
            'samples in train. This violates the standard EfficientAD anomaly '
            'detection protocol, which should train on normal samples only.'
        )

    test_labels = {
        int(mask_sum > 0)
        for mask_sum in summaries['test']['mask_sums']
    }
    if len(test_labels) < 2:
        print('WARNING: The current test split cannot reliably compute Image AUROC.')

    all_positive_pixels = sum(
        summary['mask_sum_mean'] * max(1, summary['count'])
        for summary in summaries.values()
    )
    all_pixels = 0
    for filenames in split.values():
        for filename in filenames:
            with Image.open(mask_dir / filename) as mask:
                width, height = mask.size
            all_pixels += width * height
    positive_ratio = float(all_positive_pixels) / max(1, all_pixels)
    if positive_ratio < 0.01:
        print(
            'WARNING: Pixel masks are extremely sparse. F1-max may be sensitive '
            'to class imbalance and threshold choice; inspect visualizations too.'
        )

    first_split = next((name for name in ('train', 'val', 'test') if split[name]), None)
    if first_split is None:
        raise RuntimeError('CFRP split is empty.')
    first_name = split[first_split][0]
    image = _load_rgb(image_dir / first_name)
    mask = _load_binary_mask(mask_dir / first_name)
    print(f'First sample: {first_name}')
    print(f'First image mode/size: {image.mode} {image.size}')
    print(f'First mask mode/size: {mask.mode} {mask.size}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfrp_root', default=DEFAULT_CFRP_ROOT)
    parser.add_argument('--split_path', default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    if args.check:
        check_cfrp_dataset(args.cfrp_root, args.split_path)
    else:
        build_cfrp_split(args.cfrp_root, args.split_path, write=True)
        print(f'Wrote CFRP split to {Path(args.split_path).resolve()}')


if __name__ == '__main__':
    main()
