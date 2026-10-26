from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import ImageFolderWithPath, ImageFolderWithoutTarget, InfiniteDataloader  # noqa: E402
from efficientad import (  # noqa: E402
    apply_runtime_defaults,
    build_transforms,
    collate_test_batch,
    configure_runtime,
    infer_defect_class_from_path,
    map_normalization,
    predict,
    resolve_dataset_paths,
    resolve_device,
    teacher_normalization,
)
from btad_dataset import resolve_btad_mask_path  # noqa: E402
from efficientad_variant_gs_hf_v2 import normalize_model_variant  # noqa: E402
from utils.config import load_training_config  # noqa: E402
from visa_dataset import VisaImageDataset, load_visa_records, resolve_visa_mask_path  # noqa: E402


DEFAULT_OUTPUT_ROOT = 'output/eval'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--variant', help='Optional model variant override.')
    parser.add_argument('--category', help='Optional category override.')
    parser.add_argument('--dataset_root', help='Optional dataset root override.')
    parser.add_argument(
        '--split',
        default='test_public',
        help='Evaluation split override. Defaults to test_public.',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Optional runtime device override, e.g. cuda:0 or cpu.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Optional DataLoader worker override. Defaults to the config value.',
    )
    parser.add_argument(
        '--run-dir',
        default=None,
        help='Optional specific training run directory. Defaults to latest run for variant/category.',
    )
    parser.add_argument(
        '--checkpoint-kind',
        choices=['auto', 'best', 'last'],
        default='auto',
        help='Which checkpoint bundle under the run directory to evaluate.',
    )
    parser.add_argument(
        '--output-root',
        default=DEFAULT_OUTPUT_ROOT,
        help='Root directory to store exported evaluation metrics.',
    )
    parser.add_argument(
        '--save-dir',
        default='auto',
        help='Optional absolute or output-root-relative result directory.',
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Optional train subset size for quick smoke evaluation.',
    )
    parser.add_argument(
        '--max-val-samples',
        type=int,
        default=None,
        help='Optional validation subset size for quick smoke evaluation.',
    )
    parser.add_argument(
        '--max-test-samples',
        type=int,
        default=None,
        help='Optional test subset size for quick smoke evaluation.',
    )
    return parser.parse_args()


def build_overrides(args):
    overrides = {}
    if args.variant:
        overrides.setdefault('model', {})['variant'] = args.variant
    if args.category:
        overrides.setdefault('dataset', {})['category'] = args.category
    if args.dataset_root:
        overrides.setdefault('dataset', {})['dataset_root'] = args.dataset_root
    if args.split:
        overrides.setdefault('evaluation', {})['eval_split'] = args.split
    return overrides or None


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def maybe_subset(dataset, max_samples: int | None):
    if max_samples in (None, 0):
        return dataset
    max_samples = min(int(max_samples), len(dataset))
    return Subset(dataset, list(range(max_samples)))


def find_latest_run_dir(config, explicit_run_dir: str | None) -> Path:
    if explicit_run_dir:
        run_dir = resolve_repo_path(explicit_run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f'Run directory not found: {run_dir}')
        return run_dir

    run_root = (
        Path(config.output_root)
        / str(config.model_name).strip().lower()
        / normalize_model_variant(config.variant)
        / config.category
    )
    if not run_root.is_dir():
        raise FileNotFoundError(
            'Could not find any run directory for '
            f'variant={config.variant}, category={config.category} under {run_root}.'
        )

    candidate_dirs = sorted(
        [path for path in run_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    if not candidate_dirs:
        raise FileNotFoundError(f'No run directories found under {run_root}.')
    return candidate_dirs[-1]


def resolve_checkpoint_dir(run_dir: Path, checkpoint_kind: str) -> tuple[Path, str]:
    checkpoints_root = run_dir / 'checkpoints'
    if checkpoint_kind == 'best':
        checkpoint_dir = checkpoints_root / 'best'
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_dir}')
        return checkpoint_dir, 'best'

    if checkpoint_kind == 'last':
        checkpoint_dir = checkpoints_root / 'last'
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_dir}')
        return checkpoint_dir, 'last'

    for kind in ('best', 'last'):
        checkpoint_dir = checkpoints_root / kind
        if checkpoint_dir.is_dir():
            return checkpoint_dir, kind

    raise FileNotFoundError(
        f'Could not find best/last checkpoint bundles under {checkpoints_root}.'
    )


def load_modules_from_checkpoint(checkpoint_dir: Path, device: torch.device):
    teacher = torch.load(checkpoint_dir / 'teacher.pth', map_location='cpu')
    student = torch.load(checkpoint_dir / 'student.pth', map_location='cpu')
    autoencoder = torch.load(checkpoint_dir / 'autoencoder.pth', map_location='cpu')
    evidence_fuser_path = checkpoint_dir / 'evidence_fuser.pth'
    evidence_fuser = None
    if evidence_fuser_path.is_file():
        evidence_fuser = torch.load(evidence_fuser_path, map_location='cpu')

    modules = [teacher, student, autoencoder]
    if evidence_fuser is not None:
        modules.append(evidence_fuser)
    for module in modules:
        module.to(device)
        module.eval()

    return teacher, student, autoencoder, evidence_fuser


def create_eval_dataloaders(config):
    base_transform, train_transform = build_transforms(config)
    dataset_paths = resolve_dataset_paths(config)
    pin_memory = resolve_device(config).type == 'cuda'

    if config.dataset_name == 'mvtec_ad':
        full_train_set = ImageFolderWithoutTarget(
            dataset_paths['train_path'],
            transform=train_transform,
        )
        full_train_set = maybe_subset(full_train_set, getattr(config, 'max_train_samples', None))
        test_set = ImageFolderWithPath(
            dataset_paths['test_path'],
            ignored_class_names=['ground_truth'],
        )
        if dataset_paths['validation_path'] is not None:
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                dataset_paths['validation_path'],
                transform=train_transform,
            )
        else:
            dataset_paths['validation_split'] = 'train_random_split'
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(config.seed)
            train_set, validation_set = torch.utils.data.random_split(
                full_train_set,
                [train_size, validation_size],
                rng,
            )
    elif config.dataset_name == 'btad':
        full_train_set = ImageFolderWithoutTarget(
            dataset_paths['train_path'],
            transform=train_transform,
        )
        full_train_set = maybe_subset(
            full_train_set,
            getattr(config, 'max_train_samples', None),
        )
        test_set = ImageFolderWithPath(dataset_paths['test_path'])
        dataset_paths['validation_split'] = 'train_random_split'
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(config.seed)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set,
            [train_size, validation_size],
            rng,
        )
    elif config.dataset_name == 'visa':
        train_records = load_visa_records(
            config.dataset_root,
            config.category,
            split='train',
            split_csv_file=config.split_csv_file,
        )
        if any(record.is_anomaly for record in train_records):
            raise ValueError(
                'The selected VisA split file contains anomaly samples in the '
                'training split. EfficientAD expects one-class training. '
                'Please use split_csv_file=1cls.csv.'
            )

        full_train_set = VisaImageDataset(
            config.dataset_root,
            train_records,
            transform=train_transform,
        )
        full_train_set = maybe_subset(
            full_train_set,
            getattr(config, 'max_train_samples', None),
        )
        test_records = load_visa_records(
            config.dataset_root,
            config.category,
            split=dataset_paths['test_split'],
            split_csv_file=config.split_csv_file,
        )
        test_set = VisaImageDataset(
            config.dataset_root,
            test_records,
            return_path=True,
        )
        if dataset_paths['validation_split'] is not None:
            validation_records = load_visa_records(
                config.dataset_root,
                config.category,
                split=dataset_paths['validation_split'],
                split_csv_file=config.split_csv_file,
            )
            validation_records = [
                record for record in validation_records if record.is_normal
            ]
            if validation_records:
                train_set = full_train_set
                validation_set = VisaImageDataset(
                    config.dataset_root,
                    validation_records,
                    transform=train_transform,
                )
            else:
                dataset_paths['validation_split'] = 'train_random_split'
                train_size = int(0.9 * len(full_train_set))
                validation_size = len(full_train_set) - train_size
                rng = torch.Generator().manual_seed(config.seed)
                train_set, validation_set = torch.utils.data.random_split(
                    full_train_set,
                    [train_size, validation_size],
                    rng,
                )
        else:
            dataset_paths['validation_split'] = 'train_random_split'
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(config.seed)
            train_set, validation_set = torch.utils.data.random_split(
                full_train_set,
                [train_size, validation_size],
                rng,
            )
    else:
        raise ValueError(f'Unsupported dataset_name for evaluation script: {config.dataset_name}')

    validation_set = maybe_subset(validation_set, getattr(config, 'max_val_samples', None))
    test_set = maybe_subset(test_set, getattr(config, 'max_test_samples', None))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_test_batch,
    )
    return base_transform, dataset_paths, train_loader, validation_loader, test_loader


def resolve_mask_path(image_path: str, dataset_name: str) -> Path | None:
    image_path = Path(image_path)
    if dataset_name == 'btad':
        return resolve_btad_mask_path(image_path)
    if dataset_name == 'visa':
        return resolve_visa_mask_path(image_path)

    defect_class = image_path.parent.name
    if defect_class == 'good':
        return None

    split_dir = image_path.parents[1]
    default_mask_path = split_dir / 'ground_truth' / defect_class / f'{image_path.stem}_mask.png'
    if default_mask_path.is_file():
        return default_mask_path

    fallback_matches = list((split_dir / 'ground_truth' / defect_class).glob(f'{image_path.stem}_mask.*'))
    if fallback_matches:
        return fallback_matches[0]
    raise FileNotFoundError(f'Could not find mask for image: {image_path}')


def load_binary_mask(mask_path: Path | None, width: int, height: int) -> np.ndarray:
    if mask_path is None:
        return np.zeros((height, width), dtype=np.uint8)

    mask_image = Image.open(mask_path).convert('L')
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), resample=Image.NEAREST)
    mask_array = np.array(mask_image, dtype=np.uint8)
    return (mask_array > 0).astype(np.uint8)


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true=y_true, y_score=y_score) * 100.0)


def compute_au_pro(background_chunks, region_score_chunks, fpr_limit: float = 0.3):
    if not background_chunks or not region_score_chunks:
        return None

    background_scores = np.concatenate(background_chunks).astype(np.float32, copy=False)
    num_background = int(background_scores.size)
    num_regions = int(len(region_score_chunks))
    if num_background == 0 or num_regions == 0:
        return None

    positive_scores = np.concatenate(region_score_chunks).astype(np.float32, copy=False)
    positive_weights = np.concatenate([
        np.full(chunk.size, 1.0 / (num_regions * chunk.size), dtype=np.float64)
        for chunk in region_score_chunks
    ])

    bg_unique, bg_counts = np.unique(background_scores, return_counts=True)
    pos_unique, pos_inverse = np.unique(positive_scores, return_inverse=True)
    pos_weight_sums = np.bincount(pos_inverse, weights=positive_weights)

    union_scores = np.union1d(bg_unique, pos_unique)
    bg_aligned = np.zeros(union_scores.shape[0], dtype=np.int64)
    bg_aligned[np.searchsorted(union_scores, bg_unique)] = bg_counts.astype(np.int64)
    pos_aligned = np.zeros(union_scores.shape[0], dtype=np.float64)
    pos_aligned[np.searchsorted(union_scores, pos_unique)] = pos_weight_sums

    fpr = np.cumsum(bg_aligned[::-1], dtype=np.float64) / float(num_background)
    pro = np.cumsum(pos_aligned[::-1], dtype=np.float64)
    fpr = np.concatenate(([0.0], fpr))
    pro = np.concatenate(([0.0], np.clip(pro, 0.0, 1.0)))

    cut_index = np.searchsorted(fpr, fpr_limit, side='right')
    if cut_index < len(fpr):
        previous_index = max(0, cut_index - 1)
        previous_fpr = fpr[previous_index]
        previous_pro = pro[previous_index]
        next_fpr = fpr[cut_index]
        next_pro = pro[cut_index]
        if next_fpr > previous_fpr:
            ratio = (fpr_limit - previous_fpr) / (next_fpr - previous_fpr)
            interpolated_pro = previous_pro + ratio * (next_pro - previous_pro)
        else:
            interpolated_pro = next_pro
        fpr = np.concatenate((fpr[:cut_index], [fpr_limit]))
        pro = np.concatenate((pro[:cut_index], [interpolated_pro]))
    elif fpr[-1] < fpr_limit:
        fpr = np.concatenate((fpr, [fpr_limit]))
        pro = np.concatenate((pro, [pro[-1]]))

    normalized_fpr = fpr / float(fpr_limit)
    aupro = np.trapz(pro, normalized_fpr)
    return float(np.clip(aupro, 0.0, 1.0) * 100.0)


def evaluate_checkpoint(
    config,
    device,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    train_loader,
    validation_loader,
    test_loader,
    default_transform,
):
    out_channels = config.teacher_out_channels
    teacher_mean, teacher_std = teacher_normalization(
        teacher=teacher,
        train_loader=train_loader,
        device=device,
    )
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        out_channels=out_channels,
        device=device,
        desc='Map normalization for evaluation',
    )

    image_targets = []
    image_scores = []
    pixel_target_chunks = []
    pixel_score_chunks = []
    background_chunks = []
    region_score_chunks = []
    num_positive_regions = 0
    num_defect_images = 0
    num_good_images = 0

    structure = np.ones((3, 3), dtype=np.uint8)

    with torch.inference_mode():
        for images, _, paths in tqdm(test_loader, desc='Evaluating EfficientAD'):
            original_sizes = [(image.width, image.height) for image in images]
            batch = torch.stack([default_transform(image) for image in images], dim=0)
            batch = batch.to(device)

            map_combined, _, _ = predict(
                image=batch,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                evidence_fuser=evidence_fuser,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                out_channels=out_channels,
                q_st_start=q_st_start,
                q_st_end=q_st_end,
                q_ae_start=q_ae_start,
                q_ae_end=q_ae_end,
            )

            for index, path in enumerate(paths):
                width, height = original_sizes[index]
                current_map = map_combined[index:index + 1]
                if config.pad_maps:
                    current_map = F.pad(current_map, (4, 4, 4, 4))
                current_map = F.interpolate(
                    current_map,
                    (height, width),
                    mode='bilinear',
                )
                current_map = current_map[0, 0].detach().cpu().numpy()
                current_map = np.nan_to_num(current_map, nan=0.0, posinf=0.0, neginf=0.0)

                mask_path = resolve_mask_path(path, config.dataset_name)
                current_mask = load_binary_mask(mask_path, width, height)

                defect_class = infer_defect_class_from_path(path)
                is_defect = defect_class != 'good'
                if is_defect:
                    num_defect_images += 1
                else:
                    num_good_images += 1

                image_targets.append(1 if is_defect else 0)
                image_scores.append(float(np.max(current_map)))

                pixel_target_chunks.append(current_mask.reshape(-1).astype(np.uint8))
                pixel_score_chunks.append(current_map.reshape(-1).astype(np.float32))

                positive_mask = current_mask.astype(bool)
                background_scores = current_map[~positive_mask]
                if background_scores.size:
                    background_chunks.append(background_scores.astype(np.float32, copy=False))

                labeled_mask, current_regions = ndimage.label(
                    positive_mask.astype(np.uint8),
                    structure=structure,
                )
                num_positive_regions += int(current_regions)
                for region_index in range(1, current_regions + 1):
                    region_scores = current_map[labeled_mask == region_index]
                    if region_scores.size:
                        region_score_chunks.append(region_scores.astype(np.float32, copy=False))

    pixel_targets = np.concatenate(pixel_target_chunks).astype(np.uint8, copy=False)
    pixel_scores = np.concatenate(pixel_score_chunks).astype(np.float32, copy=False)

    metrics = {
        'image_auroc': safe_roc_auc(image_targets, image_scores),
        'pixel_auroc': safe_roc_auc(pixel_targets, pixel_scores),
        'au_pro': compute_au_pro(background_chunks, region_score_chunks),
        'num_test_images': len(image_targets),
        'num_good_images': num_good_images,
        'num_defect_images': num_defect_images,
        'num_positive_pixels': int(pixel_targets.sum()),
        'num_positive_regions': num_positive_regions,
        'q_st_start': float(q_st_start.detach().cpu().item()),
        'q_st_end': float(q_st_end.detach().cpu().item()),
        'q_ae_start': float(q_ae_start.detach().cpu().item()),
        'q_ae_end': float(q_ae_end.detach().cpu().item()),
    }
    return metrics


def resolve_output_dir(args, config) -> Path:
    if args.save_dir not in (None, '', 'auto'):
        save_dir = Path(args.save_dir)
        if save_dir.is_absolute():
            return save_dir
        return (resolve_repo_path(args.output_root) / save_dir).resolve()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_root = resolve_repo_path(args.output_root)
    if config.dataset_name != 'mvtec_ad':
        output_root = output_root / config.dataset_name
    return (
        output_root
        / config.variant
        / config.category
        / timestamp
    ).resolve()


def write_csv(path: Path, payload: dict):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_summary(path: Path, payload: dict):
    lines = ['EfficientAD Evaluation Summary', '']
    for key, value in payload.items():
        lines.append(f'{key}: {value}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def format_console_summary(payload: dict) -> str:
    lines = [
        f"variant={payload['variant']}",
        f"category={payload['category']}",
        f"split={payload['split']}",
        f"image_auroc={payload['image_auroc']}",
        f"pixel_auroc={payload['pixel_auroc']}",
        f"au_pro={payload['au_pro']}",
        f"checkpoint_kind={payload['checkpoint_kind']}",
        f"checkpoint_dir={payload['checkpoint_dir']}",
        f"output_dir={payload['output_dir']}",
    ]
    return '\n'.join(lines)


def main():
    args = parse_args()
    runtime_config = load_training_config(
        str(resolve_repo_path(args.config)),
        REPO_ROOT,
        overrides=build_overrides(args),
    )
    runtime_config = apply_runtime_defaults(runtime_config)
    runtime_config.variant = normalize_model_variant(runtime_config.variant)
    runtime_config.model_variant = runtime_config.variant
    runtime_config.eval_split = args.split
    if args.device is not None:
        runtime_config.device = args.device
    if args.num_workers is not None:
        runtime_config.num_workers = int(args.num_workers)
    runtime_config.max_train_samples = args.max_train_samples
    runtime_config.max_val_samples = args.max_val_samples
    runtime_config.max_test_samples = args.max_test_samples

    configure_runtime(runtime_config.seed, runtime_config.deterministic)
    device = resolve_device(runtime_config)

    run_dir = find_latest_run_dir(runtime_config, args.run_dir)
    checkpoint_dir, checkpoint_kind = resolve_checkpoint_dir(run_dir, args.checkpoint_kind)
    teacher, student, autoencoder, evidence_fuser = load_modules_from_checkpoint(
        checkpoint_dir,
        device,
    )
    default_transform, dataset_paths, train_loader, validation_loader, test_loader = create_eval_dataloaders(
        runtime_config
    )

    metrics = evaluate_checkpoint(
        config=runtime_config,
        device=device,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        evidence_fuser=evidence_fuser,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        default_transform=default_transform,
    )

    output_dir = resolve_output_dir(args, runtime_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolve_repo_path(args.config)
    shutil.copy2(config_path, output_dir / f'input_config{config_path.suffix}')
    run_summary_path = run_dir / 'run_summary.json'
    if run_summary_path.is_file():
        shutil.copy2(run_summary_path, output_dir / 'source_run_summary.json')

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'variant': runtime_config.variant,
        'category': runtime_config.category,
        'split': runtime_config.eval_split,
        'device': str(device),
        'config_path': str(config_path),
        'dataset_root': runtime_config.dataset_root,
        'resolved_test_path': dataset_paths['test_path'],
        'split_csv_path': dataset_paths.get('split_csv_path'),
        'run_dir': str(run_dir),
        'checkpoint_kind': checkpoint_kind,
        'checkpoint_dir': str(checkpoint_dir),
        'output_dir': str(output_dir),
        'image_auroc': metrics['image_auroc'],
        'pixel_auroc': metrics['pixel_auroc'],
        'au_pro': metrics['au_pro'],
        'num_test_images': metrics['num_test_images'],
        'num_good_images': metrics['num_good_images'],
        'num_defect_images': metrics['num_defect_images'],
        'num_positive_pixels': metrics['num_positive_pixels'],
        'num_positive_regions': metrics['num_positive_regions'],
        'q_st_start': metrics['q_st_start'],
        'q_st_end': metrics['q_st_end'],
        'q_ae_start': metrics['q_ae_start'],
        'q_ae_end': metrics['q_ae_end'],
    }

    write_csv(output_dir / 'metrics.csv', payload)
    write_json(output_dir / 'metrics.json', payload)
    write_summary(output_dir / 'summary.txt', payload)

    print(format_console_summary(payload))


if __name__ == '__main__':
    main()
