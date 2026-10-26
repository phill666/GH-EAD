from __future__ import annotations

from pathlib import Path


BTAD_GOOD_CLASS_NAMES = {'good', 'ok', 'normal'}
DEFAULT_BTAD_ROOT = 'data/btad'


def resolve_btad_layout(dataset_root: str | Path, category: str) -> dict[str, str]:
    dataset_root = Path(dataset_root)
    candidate_roots = [
        (dataset_root, dataset_root / category),
        (dataset_root.parent, dataset_root),
    ]

    checked = []
    for root_candidate, category_candidate in candidate_roots:
        train_path = category_candidate / 'train'
        test_path = category_candidate / 'test'
        ground_truth_path = category_candidate / 'ground_truth'
        checked.append(
            {
                'dataset_root': str(root_candidate),
                'category_root': str(category_candidate),
            }
        )
        if train_path.is_dir() and test_path.is_dir() and ground_truth_path.is_dir():
            return {
                'dataset_root': str(root_candidate.resolve()),
                'category_root': str(category_candidate.resolve()),
                'train_path': str(train_path.resolve()),
                'test_path': str(test_path.resolve()),
                'ground_truth_path': str(ground_truth_path.resolve()),
            }

    raise FileNotFoundError(
        'Could not resolve BTAD dataset root/category root from '
        f'dataset_root={dataset_root} and category={category}. Checked: {checked}'
    )


def normalize_btad_defect_class(defect_class: str) -> str:
    normalized = str(defect_class).strip().lower()
    return 'good' if normalized in BTAD_GOOD_CLASS_NAMES else normalized


def infer_btad_defect_class(image_path: str | Path) -> str:
    image_path = Path(image_path)
    return normalize_btad_defect_class(image_path.parent.name)


def resolve_btad_mask_path(image_path: str | Path) -> Path | None:
    image_path = Path(image_path)
    defect_class = normalize_btad_defect_class(image_path.parent.name)
    if defect_class == 'good':
        return None

    category_root = image_path.parents[2]
    mask_dir = category_root / 'ground_truth' / image_path.parent.name
    if not mask_dir.is_dir():
        fallback_dir = category_root / 'ground_truth' / defect_class
        if fallback_dir.is_dir():
            mask_dir = fallback_dir
        else:
            raise FileNotFoundError(f'Could not find BTAD mask directory for image: {image_path}')

    exact_mask_path = mask_dir / image_path.name
    if exact_mask_path.is_file():
        return exact_mask_path

    png_mask_path = mask_dir / f'{image_path.stem}.png'
    if png_mask_path.is_file():
        return png_mask_path

    bmp_mask_path = mask_dir / f'{image_path.stem}.bmp'
    if bmp_mask_path.is_file():
        return bmp_mask_path

    fallback_matches = sorted(mask_dir.glob(f'{image_path.stem}.*'))
    if fallback_matches:
        return fallback_matches[0]

    raise FileNotFoundError(f'Could not find BTAD mask for image: {image_path}')
