import argparse
import csv
import itertools
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    from sklearn.metrics import precision_recall_curve, roc_auc_score
except ImportError:
    precision_recall_curve = None
    roc_auc_score = None

from common import (
    ImageFolderWithPath,
    ImageFolderWithoutTarget,
    InfiniteDataloader,
    get_pdn_medium,
    get_pdn_small,
)
from efficientad_variant_gs_hf_v2 import (
    BASELINE_VARIANT,
    GS_AE_HF_VARIANT,
    GS_AE_ONLY_VARIANT,
    GS_HF_FULL_VARIANT,
    GS_STUDENT_CONV1_HF_VARIANT,
    GS_STUDENT_CONV1_ONLY_VARIANT,
    GS_STUDENT_CONV2_ONLY_VARIANT,
    GS_STUDENT_ONLY_VARIANT,
    build_autoencoder_model,
    build_evidence_fuser,
    build_student_model,
    normalize_model_variant,
)
from btad_dataset import (
    DEFAULT_BTAD_ROOT,
    infer_btad_defect_class,
    normalize_btad_defect_class,
    resolve_btad_layout,
)
from cfrp_dataset import (
    DEFAULT_CFRP_ROOT,
    DEFAULT_SPLIT_PATH as DEFAULT_CFRP_SPLIT_PATH,
    CfrpEvalDataset,
    CfrpTrainDataset,
    collate_cfrp_eval_batch,
    load_or_create_cfrp_split,
)
from visa_dataset import (
    DEFAULT_VISA_SPLIT_CSV,
    VisaImageDataset,
    infer_visa_defect_class,
    load_visa_records,
    resolve_visa_layout,
)

try:
    from utils.config import format_config_summary, load_training_config
except ImportError:
    format_config_summary = None
    load_training_config = None


DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_NUM_WORKERS = 4
DEFAULT_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


class TrainPairTransform:
    """Pickle-friendly train transform for Windows DataLoader workers."""

    def __init__(self, base_transform, ae_augmentation):
        self.base_transform = base_transform
        self.ae_augmentation = ae_augmentation

    def __call__(self, image):
        return (
            self.base_transform(image),
            self.base_transform(self.ae_augmentation(image)),
        )


def load_json_cli_defaults(config_path, valid_keys):
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data = json.load(file)

    if not isinstance(config_data, dict):
        raise ValueError('Flat JSON config must contain a JSON object.')

    unknown_keys = sorted(set(config_data) - valid_keys)
    if unknown_keys:
        raise ValueError(
            'Unknown JSON config keys: ' + ', '.join(unknown_keys)
        )
    return config_data


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='Path to a YAML config file or a flat JSON config file.',
    )
    parser.add_argument(
        '-d',
        '--dataset',
        default='mvtec_ad',
        choices=['mvtec_ad', 'mvtec_loco', 'visa', 'btad', 'cfrp'],
    )
    parser.add_argument(
        '-s',
        '--subdataset',
        default='bottle',
        help='One of the sub-datasets of MVTec AD or MVTec LOCO.',
    )
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument(
        '--save_dir',
        default=None,
        help='Lightweight output directory used by the CFRP train/test mode.',
    )
    parser.add_argument(
        '-m',
        '--model_size',
        default='small',
        choices=['small', 'medium', 's', 'm'],
    )
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument(
        '-i',
        '--imagenet_train_path',
        default='none',
        help='Set to "none" to disable the ImageNet pretraining penalty.',
    )
    parser.add_argument(
        '-a',
        '--mvtec_ad_path',
        default='./mvtec_anomaly_detection',
        help='Root path of the downloaded MVTec AD dataset.',
    )
    parser.add_argument(
        '-b',
        '--mvtec_loco_path',
        default='./mvtec_loco_anomaly_detection',
        help='Root path of the downloaded MVTec LOCO dataset.',
    )
    parser.add_argument(
        '--cfrp_root',
        default=DEFAULT_CFRP_ROOT,
        help='Root path of the CFRP infrared image/mask dataset.',
    )
    parser.add_argument('-t', '--train_steps', '--steps', type=int, default=70000)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        '--teacher_out_channels',
        type=int,
        default=384,
    )
    parser.add_argument(
        '--model_variant',
        default=BASELINE_VARIANT,
        choices=[
            BASELINE_VARIANT,
            GS_STUDENT_ONLY_VARIANT,
            GS_AE_ONLY_VARIANT,
            GS_AE_HF_VARIANT,
            GS_STUDENT_CONV1_HF_VARIANT,
            GS_STUDENT_CONV1_ONLY_VARIANT,
            GS_STUDENT_CONV2_ONLY_VARIANT,
            GS_HF_FULL_VARIANT,
        ],
    )
    parser.add_argument('--hf_fusion_beta', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eval_split', default='test_public')
    parser.add_argument('--padding', action='store_true')
    parser.add_argument('--pad_maps', dest='pad_maps', action='store_true')
    parser.add_argument('--no_pad_maps', dest='pad_maps', action='store_false')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--check_val_every_n_steps', type=int, default=10000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None)
    parser.add_argument('--save_every_n_steps', type=int, default=1000)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save_vis', type=int, default=20)

    parser.set_defaults(pad_maps=True)

    valid_keys = {action.dest for action in parser._actions}
    initial_args, _ = parser.parse_known_args()
    if initial_args.config:
        suffix = Path(initial_args.config).suffix.lower()
        if suffix == '.json':
            config_defaults = load_json_cli_defaults(
                initial_args.config,
                valid_keys,
            )
            parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    if args.config:
        suffix = Path(args.config).suffix.lower()
        if suffix in {'.yaml', '.yml'}:
            if load_training_config is None:
                raise RuntimeError(
                    'YAML config support requires utils.config to be present.'
                )
            repo_root = Path(__file__).resolve().parent
            config_path = args.config
            if not os.path.isabs(config_path):
                config_path = str((repo_root / config_path).resolve())
            return apply_runtime_defaults(
                load_training_config(config_path, repo_root)
            )
    return apply_runtime_defaults(args)


def normalize_model_size(model_size):
    normalized = str(model_size).strip().lower()
    if normalized in {'s', 'small'}:
        return 'small'
    if normalized in {'m', 'medium'}:
        return 'medium'
    raise ValueError(f'Unsupported model_size: {model_size}')


def apply_runtime_defaults(config):
    if hasattr(config, 'dataset') and not hasattr(config, 'dataset_name'):
        setattr(config, 'dataset_name', getattr(config, 'dataset'))
    if hasattr(config, 'subdataset') and not hasattr(config, 'category'):
        setattr(config, 'category', getattr(config, 'subdataset'))
    if hasattr(config, 'model_variant') and not hasattr(config, 'variant'):
        setattr(config, 'variant', getattr(config, 'model_variant'))
    if hasattr(config, 'variant') and not hasattr(config, 'model_variant'):
        setattr(config, 'model_variant', getattr(config, 'variant'))
    dataset_name = getattr(config, 'dataset_name', getattr(config, 'dataset', 'mvtec_ad'))
    if dataset_name == 'cfrp' and getattr(config, 'save_dir', None) not in (None, '', 'auto'):
        setattr(config, 'output_dir', getattr(config, 'save_dir'))

    dataset_root = getattr(config, 'dataset_root', None)
    if dataset_root is None:
        if dataset_name == 'mvtec_ad':
            dataset_root = getattr(
                config,
                'mvtec_ad_path',
                './mvtec_anomaly_detection',
            )
        elif dataset_name == 'btad':
            dataset_root = getattr(
                config,
                'btad_path',
                DEFAULT_BTAD_ROOT,
            )
        elif dataset_name == 'visa':
            dataset_root = getattr(
                config,
                'visa_path',
                './Visa',
            )
        elif dataset_name == 'cfrp':
            dataset_root = getattr(
                config,
                'cfrp_root',
                DEFAULT_CFRP_ROOT,
            )
        else:
            dataset_root = getattr(
                config,
                'mvtec_loco_path',
                './mvtec_loco_anomaly_detection',
            )
        setattr(config, 'dataset_root', dataset_root)

    if not hasattr(config, 'weights') and hasattr(config, 'pretrained_teacher'):
        setattr(config, 'weights', getattr(config, 'pretrained_teacher'))
    if not hasattr(config, 'pretrained_teacher') and hasattr(config, 'weights'):
        setattr(config, 'pretrained_teacher', getattr(config, 'weights'))

    if not hasattr(config, 'imagenet_train_path') and hasattr(config, 'imagenet_dir'):
        setattr(config, 'imagenet_train_path', getattr(config, 'imagenet_dir'))
    if not hasattr(config, 'imagenet_dir') and hasattr(config, 'imagenet_train_path'):
        setattr(config, 'imagenet_dir', getattr(config, 'imagenet_train_path'))

    if not hasattr(config, 'normalization') or getattr(config, 'normalization') is None:
        setattr(config, 'normalization', dict(DEFAULT_NORMALIZATION))
    if not hasattr(config, 'center_crop'):
        setattr(config, 'center_crop', None)
    if not hasattr(config, 'device') or getattr(config, 'device') in (None, '', 'auto'):
        default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        setattr(config, 'device', default_device)

    defaults = {
        'config': None,
        'experiment_name': 'efficientad_cli',
        'project_name': 'EfficientAD',
        'output_root': str(
            Path(getattr(config, 'output_dir', 'output')).resolve().parent
        ),
        'save_dir': 'auto',
        'num_workers': DEFAULT_NUM_WORKERS,
        'split_csv_file': DEFAULT_VISA_SPLIT_CSV,
        'train_dir_name': 'train',
        'val_dir_name': 'validation',
        'test_dir_name': 'test',
        'use_validation': True,
        'use_test_public': True,
        'support_test_private': False,
        'batch_size': DEFAULT_BATCH_SIZE,
        'val_batch_size': getattr(config, 'batch_size', DEFAULT_BATCH_SIZE),
        'test_batch_size': getattr(config, 'batch_size', DEFAULT_BATCH_SIZE),
        'model_name': 'EfficientAD',
        'variant': BASELINE_VARIANT,
        'model_variant': BASELINE_VARIANT,
        'hf_fusion_beta': 0.2,
        'model_size': getattr(config, 'model_size', 'small'),
        'teacher_out_channels': 384,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'weight_decay': DEFAULT_WEIGHT_DECAY,
        'padding': False,
        'pad_maps': True,
        'use_pretrained_teacher': True,
        'train_steps': 70000,
        'max_epochs': None,
        'accelerator': 'cuda'
        if str(getattr(config, 'device', 'cuda:0')).startswith('cuda')
        else 'cpu',
        'precision': 32,
        'gradient_clip_val': 0.0,
        'accumulate_grad_batches': 1,
        'log_every_n_steps': 10,
        'check_val_every_n_epoch': None,
        'check_val_every_n_steps': 10000,
        'save_every_n_steps': 1000,
        'deterministic': False,
        'save_top_k': 1,
        'monitor_metric': 'image_auc',
        'monitor_mode': 'max',
        'save_last': True,
        'enable_checkpointing': True,
        'enable_tensorboard': False,
        'enable_csv_logger': True,
        'save_predictions': True,
        'save_visualizations': False,
        'run_test_after_fit': True,
        'eval_split': 'test_public',
        'metrics': ['image_auc'],
        'threshold': None,
        'save_anomaly_maps': True,
        'raw_config': None,
        'mode': 'train',
        'checkpoint': None,
        'save_vis': 20,
    }

    for name, value in defaults.items():
        if not hasattr(config, name) or getattr(config, name) is None:
            setattr(config, name, value)

    config.model_size = normalize_model_size(getattr(config, 'model_size'))
    config.variant = normalize_model_variant(getattr(config, 'variant', BASELINE_VARIANT))
    config.model_variant = config.variant
    config.hf_fusion_beta = float(getattr(config, 'hf_fusion_beta', 0.2))
    if getattr(config, 'val_batch_size') in (None, 0):
        setattr(config, 'val_batch_size', getattr(config, 'batch_size'))
    if getattr(config, 'test_batch_size') in (None, 0):
        setattr(config, 'test_batch_size', getattr(config, 'batch_size'))

    return config


def configure_runtime(seed, deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(config):
    requested_device = str(config.device).strip().lower()
    requested_accelerator = str(getattr(config, 'accelerator', '')).strip().lower()

    if requested_device.startswith('cuda') or requested_accelerator == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'Config requested CUDA ({config.device}) but torch.cuda.is_available() is False.'
            )
        return torch.device(config.device)

    if requested_device == 'cpu' or requested_accelerator == 'cpu':
        return torch.device('cpu')

    if requested_device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return torch.device(config.device)


def build_transforms(config):
    normalization = config.normalization or DEFAULT_NORMALIZATION
    base_ops = [transforms.Resize((config.image_size, config.image_size))]
    if config.center_crop:
        base_ops.append(transforms.CenterCrop(config.center_crop))
    base_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalization['mean'],
            std=normalization['std'],
        ),
    ])
    base_transform = transforms.Compose(base_ops)
    transform_ae = transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2),
    ])
    train_transform = TrainPairTransform(base_transform, transform_ae)
    return base_transform, train_transform


def resolve_existing_dir(base_path, candidate_names, required=True):
    for candidate_name in candidate_names:
        if candidate_name in (None, ''):
            continue
        candidate_path = os.path.join(base_path, candidate_name)
        if os.path.isdir(candidate_path):
            return candidate_name, candidate_path

    if required:
        joined_candidates = ', '.join(str(item) for item in candidate_names if item)
        raise FileNotFoundError(
            f'Could not find any of [{joined_candidates}] under {base_path}.'
        )
    return None, None


def deduplicate_preserve_order(values):
    deduplicated = []
    for value in values:
        if value and value not in deduplicated:
            deduplicated.append(value)
    return deduplicated


def resolve_category_root(dataset_root, category, train_dir_name):
    candidate_roots = [
        Path(dataset_root),
        Path(dataset_root) / category,
    ]
    for candidate in candidate_roots:
        if (candidate / train_dir_name).is_dir():
            return str(candidate.resolve())
    raise FileNotFoundError(
        'Could not resolve dataset category root. Tried: '
        + ', '.join(str(candidate) for candidate in candidate_roots)
    )


def resolve_dataset_paths(config):
    if (
        not config.support_test_private
        and config.eval_split in {'test_private', 'test_private_mixed'}
    ):
        raise ValueError(
            'evaluation split points to a private split, but support_test_private is false.'
        )

    if config.dataset_name == 'visa':
        layout = resolve_visa_layout(
            config.dataset_root,
            config.category,
            split_csv_file=getattr(config, 'split_csv_file', DEFAULT_VISA_SPLIT_CSV),
        )
        config.dataset_root = layout['dataset_root']
        category_root = layout['category_root']
        split_csv_path = layout['split_csv_path']
        records = load_visa_records(
            config.dataset_root,
            config.category,
            split_csv_file=getattr(config, 'split_csv_file', DEFAULT_VISA_SPLIT_CSV),
        )
        available_splits = {record.split for record in records}
        if 'train' not in available_splits:
            raise FileNotFoundError(
                f'VisA split CSV does not contain a train split for {config.category}.'
            )

        validation_split = None
        if config.use_validation:
            for split_name in ('validation', 'val'):
                if split_name in available_splits:
                    validation_split = split_name
                    break

        requested_eval_split = str(getattr(config, 'eval_split', 'test') or 'test')
        if requested_eval_split in {'test', 'test_public'}:
            test_split = 'test'
        elif requested_eval_split in available_splits:
            test_split = requested_eval_split
        else:
            raise ValueError(
                f'Unsupported VisA eval split "{requested_eval_split}". '
                f'Available splits for {config.category}: {sorted(available_splits)}'
            )

        images_root = str((Path(category_root) / 'Data' / 'Images').resolve())
        return {
            'category_root': category_root,
            'split_csv_path': split_csv_path,
            'available_splits': sorted(available_splits),
            'train_split': 'train',
            'train_path': images_root,
            'validation_split': validation_split,
            'validation_path': images_root if validation_split is not None else None,
            'test_split': test_split,
            'test_path': images_root,
        }

    if config.dataset_name == 'btad':
        layout = resolve_btad_layout(config.dataset_root, config.category)
        config.dataset_root = layout['dataset_root']
        category_root = layout['category_root']
        validation_split = None
        test_candidates = deduplicate_preserve_order([
            config.eval_split if getattr(config, 'eval_split', None) else None,
            config.test_dir_name,
            'test',
        ])
        if 'test' not in test_candidates:
            raise ValueError(
                f'Unsupported BTAD eval split "{config.eval_split}". '
                'BTAD currently exposes train/test directories only.'
            )
        return {
            'category_root': category_root,
            'ground_truth_path': layout['ground_truth_path'],
            'train_split': 'train',
            'train_path': layout['train_path'],
            'validation_split': validation_split,
            'validation_path': None,
            'test_split': 'test',
            'test_path': layout['test_path'],
        }

    category_root = resolve_category_root(
        config.dataset_root,
        config.category,
        config.train_dir_name,
    )
    train_split, train_path = resolve_existing_dir(
        category_root,
        [config.train_dir_name],
    )

    validation_split = None
    validation_path = None
    if config.dataset_name == 'mvtec_ad':
        if config.use_validation:
            validation_split, validation_path = resolve_existing_dir(
                category_root,
                [config.val_dir_name],
                required=False,
            )

        test_candidates = [config.eval_split]
        if config.use_test_public:
            test_candidates.append('test_public')
        test_candidates.append(config.test_dir_name)
        if config.support_test_private:
            test_candidates.extend(['test_private', 'test_private_mixed'])
        test_split, test_path = resolve_existing_dir(
            category_root,
            deduplicate_preserve_order(test_candidates),
        )
    elif config.dataset_name == 'mvtec_loco':
        validation_split, validation_path = resolve_existing_dir(
            category_root,
            [config.val_dir_name],
            required=False,
        )
        test_split, test_path = resolve_existing_dir(
            category_root,
            deduplicate_preserve_order([config.eval_split, config.test_dir_name, 'test']),
        )
    else:
        raise ValueError(f'Unsupported dataset_name: {config.dataset_name}')

    return {
        'category_root': category_root,
        'train_split': train_split,
        'train_path': train_path,
        'validation_split': validation_split,
        'validation_path': validation_path,
        'test_split': test_split,
        'test_path': test_path,
    }


def create_output_dirs(config, test_split):
    output_dir = Path(config.output_dir)
    train_output_dir = output_dir / 'trainings' / config.dataset_name / config.category
    checkpoints_dir = output_dir / 'checkpoints'
    step_checkpoints_dir = checkpoints_dir / 'steps'
    best_checkpoint_dir = checkpoints_dir / 'best'
    last_checkpoint_dir = checkpoints_dir / 'last'
    logs_dir = output_dir / 'logs'
    test_output_dir = (
        output_dir
        / 'anomaly_maps'
        / config.dataset_name
        / config.category
        / test_split
    )
    predictions_dir = (
        output_dir
        / 'predictions'
        / config.dataset_name
        / config.category
        / test_split
    )
    visualizations_dir = (
        output_dir
        / 'visualizations'
        / config.dataset_name
        / config.category
        / test_split
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if config.enable_checkpointing:
        step_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if config.save_top_k:
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if config.save_last:
            last_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.save_anomaly_maps:
        test_output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    if config.save_visualizations:
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    return {
        'output_dir': str(output_dir),
        'train_output_dir': str(train_output_dir),
        'checkpoints_dir': str(checkpoints_dir),
        'step_checkpoints_dir': str(step_checkpoints_dir),
        'best_checkpoint_dir': str(best_checkpoint_dir),
        'last_checkpoint_dir': str(last_checkpoint_dir),
        'logs_dir': str(logs_dir),
        'metrics_csv_path': str(logs_dir / 'metrics.csv'),
        'predictions_csv_path': str(predictions_dir / 'predictions.csv'),
        'test_output_dir': str(test_output_dir),
        'visualizations_dir': str(visualizations_dir),
        'run_summary_path': str(output_dir / 'run_summary.json'),
        'resolved_config_path': str(output_dir / 'resolved_config.json'),
        'run_log_path': str(output_dir / 'run.log'),
    }


def create_metric_logger(config, paths):
    if not config.enable_csv_logger:
        return None

    csv_path = Path(paths['metrics_csv_path'])
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['stage', 'step', 'metric', 'value'],
            )
            writer.writeheader()

    def log_metrics(stage, step, metrics):
        with open(csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['stage', 'step', 'metric', 'value'],
            )
            for metric_name, metric_value in metrics.items():
                writer.writerow({
                    'stage': stage,
                    'step': step,
                    'metric': metric_name,
                    'value': metric_value,
                })

    return log_metrics


def create_tensorboard_writer(config, paths):
    if not config.enable_tensorboard:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            'enable_tensorboard is true, but torch.utils.tensorboard is not available.'
        ) from exc

    tb_dir = Path(paths['logs_dir']) / 'tensorboard'
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def log_run_message(paths, message, print_message=True):
    if print_message:
        print(message)
    with open(paths['run_log_path'], 'a', encoding='utf-8') as file:
        file.write(f'{message}\n')


def save_runtime_config_snapshot(config, paths):
    source_config_path = getattr(config, 'config', None)
    if source_config_path and Path(source_config_path).is_file():
        source_path = Path(source_config_path)
        copied_name = f'input_config{source_path.suffix or ".txt"}'
        shutil.copy2(source_path, Path(paths['output_dir']) / copied_name)

    payload = getattr(config, 'raw_config', None)
    if payload is None:
        payload = {
            key: value
            for key, value in vars(config).items()
            if key != 'raw_config'
        }
    with open(paths['resolved_config_path'], 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def log_metrics(log_fn, writer, stage, step, metrics):
    if log_fn is not None:
        log_fn(stage, step, metrics)
    if writer is not None:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f'{stage}/{metric_name}', metric_value, step)


def normalize_anomaly_map(anomaly_map, q_start=None, q_end=None, scale=0.1):
    normalized = anomaly_map
    if q_start is not None and q_end is not None:
        eps = torch.finfo(normalized.dtype).eps
        denominator = torch.clamp(q_end - q_start, min=eps)
        normalized = scale * (normalized - q_start) / denominator
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def hard_feature_loss(distance, hard_fraction=0.001):
    flattened = distance.reshape(-1)
    k = max(1, int(math.ceil(flattened.numel() * hard_fraction)))
    return torch.topk(flattened, k, largest=True).values.mean()


def collate_test_batch(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def remove_path_if_exists(path):
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def save_torch_object_atomic(obj, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f'{target_path.name}.tmp')
    remove_path_if_exists(temp_path)
    try:
        torch.save(obj, temp_path)
        if target_path.exists():
            remove_path_if_exists(target_path)
        temp_path.replace(target_path)
    except Exception:
        remove_path_if_exists(temp_path)
        raise


def save_checkpoint_bundle(
    target_dir,
    teacher,
    student,
    autoencoder,
    metadata,
    extra_modules=None,
):
    target_path = Path(target_dir)
    temp_path = target_path.with_name(f'{target_path.name}.tmp')
    remove_path_if_exists(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    try:
        save_torch_object_atomic(teacher, temp_path / 'teacher.pth')
        save_torch_object_atomic(student, temp_path / 'student.pth')
        save_torch_object_atomic(autoencoder, temp_path / 'autoencoder.pth')
        for module_name, module in (extra_modules or {}).items():
            if module is None:
                continue
            save_torch_object_atomic(module, temp_path / f'{module_name}.pth')
        with open(temp_path / 'metadata.json', 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2)
        if target_path.exists():
            remove_path_if_exists(target_path)
        temp_path.replace(target_path)
    except Exception:
        remove_path_if_exists(temp_path)
        raise


def save_visualization_image(anomaly_map, file_path):
    anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(anomaly_map))
    maximum = float(np.max(anomaly_map))
    if maximum > minimum:
        normalized = (anomaly_map - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(anomaly_map)
    image = Image.fromarray(
        np.uint8(np.clip(normalized, 0.0, 1.0) * 255),
        mode='L',
    )
    image.save(file_path)


def write_predictions_csv(csv_path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_models(config, device):
    out_channels = config.teacher_out_channels
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels, padding=config.padding)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels, padding=config.padding)
    else:
        raise ValueError(f'Unsupported model_size: {config.model_size}')

    student = build_student_model(
        model_size=config.model_size,
        out_channels=out_channels,
        padding=config.padding,
        variant=config.variant,
    )

    if config.use_pretrained_teacher:
        weights_path = Path(config.pretrained_teacher)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f'Could not find pretrained teacher weights: {weights_path}'
            )
        state_dict = torch.load(weights_path, map_location='cpu')
        teacher.load_state_dict(state_dict)

    autoencoder = build_autoencoder_model(out_channels, config.variant)
    evidence_fuser = build_evidence_fuser(config.variant, config.hf_fusion_beta)
    teacher.to(device)
    student.to(device)
    autoencoder.to(device)
    if evidence_fuser is not None:
        evidence_fuser.to(device)
    return teacher, student, autoencoder, evidence_fuser


def resolve_train_steps(config, train_loader):
    if config.train_steps not in (None, 0):
        return int(config.train_steps)
    if config.max_epochs not in (None, 0):
        steps = int(
            math.ceil(float(config.max_epochs) * max(1, len(train_loader)))
        )
        if steps <= 0:
            raise ValueError('Resolved train_steps must be positive.')
        return steps
    raise ValueError('Either train_steps or max_epochs must be configured.')


def resolve_eval_interval(config, train_loader_length):
    if config.check_val_every_n_steps not in (None, 0):
        return int(config.check_val_every_n_steps)
    if config.check_val_every_n_epoch not in (None, 0):
        return int(config.check_val_every_n_epoch) * max(1, train_loader_length)
    return None


def is_better(metric_value, best_value, mode):
    if best_value is None:
        return True
    if mode == 'max':
        return metric_value > best_value
    return metric_value < best_value


def print_runtime_summary(
    config,
    paths,
    dataset_paths,
    resolved_train_steps,
    device,
    train_loader_length,
):
    summary = {
        'config_path': config.config,
        'output_dir': paths['output_dir'],
        'dataset_name': config.dataset_name,
        'category': config.category,
        'category_root': dataset_paths['category_root'],
        'model_variant': config.variant,
        'train_path': dataset_paths['train_path'],
        'validation_path': dataset_paths['validation_path'],
        'test_path': dataset_paths['test_path'],
        'train_steps': resolved_train_steps,
        'max_epochs': config.max_epochs,
        'effective_epochs': (
            resolved_train_steps / max(1, train_loader_length)
        ),
        'image_size': config.image_size,
        'batch_size': config.batch_size,
        'val_batch_size': config.val_batch_size,
        'test_batch_size': config.test_batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'device': str(device),
        'model_size': config.model_size,
        'teacher_out_channels': config.teacher_out_channels,
        'padding': config.padding,
        'pad_maps': config.pad_maps,
        'eval_split': dataset_paths['test_split'],
        'split_csv_file': getattr(config, 'split_csv_file', None),
        'split_csv_path': dataset_paths.get('split_csv_path'),
    }
    print('Runtime summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_prediction_rows(y_true, y_score, paths, threshold):
    rows = []
    for current_y_true, current_y_score, path in zip(y_true, y_score, paths):
        defect_class = infer_defect_class_from_path(path)
        row = {
            'path': path,
            'defect_class': defect_class,
            'label': current_y_true,
            'score': current_y_score,
        }
        if threshold is not None:
            row['threshold'] = threshold
            row['pred_label'] = int(current_y_score >= threshold)
        rows.append(row)
    return rows


def infer_defect_class_from_path(path):
    path = Path(path)
    if 'Data' in path.parts and 'Images' in path.parts:
        return infer_visa_defect_class(path)
    if 'ground_truth' in path.parts or 'test' in path.parts or 'train' in path.parts:
        return normalize_btad_defect_class(path.parent.name)
    return normalize_btad_defect_class(os.path.basename(os.path.dirname(path)))


def infer_image_label_from_path(path):
    return 0 if infer_defect_class_from_path(path) == 'good' else 1


def evaluate_and_maybe_checkpoint(
    step,
    config,
    device,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    teacher_mean,
    teacher_std,
    validation_loader,
    test_loader,
    dataset_paths,
    paths,
    default_transform,
    log_fn,
    writer,
    best_metric_value,
    best_step,
    desc_prefix,
):
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        out_channels=config.teacher_out_channels,
        device=device,
        desc=f'{desc_prefix} map normalization',
    )
    auc, prediction_rows = test(
        test_loader=test_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        evidence_fuser=evidence_fuser,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        out_channels=config.teacher_out_channels,
        device=device,
        default_transform=default_transform,
        q_st_start=q_st_start,
        q_st_end=q_st_end,
        q_ae_start=q_ae_start,
        q_ae_end=q_ae_end,
        test_output_dir=paths['test_output_dir'] if config.save_anomaly_maps else None,
        predictions_csv_path=(
            paths['predictions_csv_path'] if config.save_predictions else None
        ),
        visualizations_dir=(
            paths['visualizations_dir'] if config.save_visualizations else None
        ),
        save_anomaly_maps=config.save_anomaly_maps,
        save_visualizations=config.save_visualizations,
        pad_maps=config.pad_maps,
        threshold=config.threshold,
        dataset_name=config.dataset_name,
        desc=f'{desc_prefix} inference',
    )

    metrics = {'image_auc': auc}
    log_metrics(log_fn, writer, 'eval', step, metrics)

    if config.enable_checkpointing and config.save_top_k:
        if is_better(auc, best_metric_value, config.monitor_mode):
            metadata = {
                'step': step,
                'monitor_metric': config.monitor_metric,
                'monitor_mode': config.monitor_mode,
                'metric_value': auc,
                'test_split': dataset_paths['test_split'],
                'predictions_csv_path': (
                    paths['predictions_csv_path'] if config.save_predictions else None
                ),
                'num_predictions': len(prediction_rows),
            }
            save_checkpoint_bundle(
                paths['best_checkpoint_dir'],
                teacher,
                student,
                autoencoder,
                metadata,
                extra_modules={'evidence_fuser': evidence_fuser},
            )
            best_metric_value = auc
            best_step = step

    return {
        'auc': auc,
        'best_metric_value': best_metric_value,
        'best_step': best_step,
        'q_st_start': q_st_start,
        'q_st_end': q_st_end,
        'q_ae_start': q_ae_start,
        'q_ae_end': q_ae_end,
    }


def run_training(config):
    config = apply_runtime_defaults(config)

    configure_runtime(config.seed, config.deterministic)
    device = resolve_device(config)
    pin_memory = device.type == 'cuda'
    base_transform, train_transform = build_transforms(config)

    dataset_paths = resolve_dataset_paths(config)
    category_root = dataset_paths['category_root']
    train_path = dataset_paths['train_path']
    validation_path = dataset_paths['validation_path']
    test_path = dataset_paths['test_path']
    test_split = dataset_paths['test_split']

    if config.dataset_name == 'mvtec_ad':
        full_train_set = ImageFolderWithoutTarget(
            train_path,
            transform=train_transform,
        )
        test_set = ImageFolderWithPath(
            test_path,
            ignored_class_names=['ground_truth'],
        )
        if validation_path is not None:
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                validation_path,
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
            train_path,
            transform=train_transform,
        )
        test_set = ImageFolderWithPath(test_path)
        dataset_paths['validation_split'] = 'train_random_split'
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(config.seed)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set,
            [train_size, validation_size],
            rng,
        )
    elif config.dataset_name == 'mvtec_loco':
        full_train_set = ImageFolderWithoutTarget(
            train_path,
            transform=train_transform,
        )
        test_set = ImageFolderWithPath(test_path)
        train_set = full_train_set
        if validation_path is None:
            raise FileNotFoundError(
                f'Could not find validation split under {category_root}.'
            )
        validation_set = ImageFolderWithoutTarget(
            validation_path,
            transform=train_transform,
        )
    elif config.dataset_name == 'visa':
        train_records = load_visa_records(
            config.dataset_root,
            config.category,
            split='train',
            split_csv_file=config.split_csv_file,
        )
        if not train_records:
            raise FileNotFoundError(
                f'Could not find VisA train records for category {config.category}.'
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
        test_records = load_visa_records(
            config.dataset_root,
            config.category,
            split=test_split,
            split_csv_file=config.split_csv_file,
        )
        test_set = VisaImageDataset(
            config.dataset_root,
            test_records,
            return_path=True,
        )

        if validation_path is not None and dataset_paths['validation_split'] is not None:
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
        raise ValueError(f'Unsupported dataset_name: {config.dataset_name}')

    paths = create_output_dirs(config, test_split)
    save_runtime_config_snapshot(config, paths)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    if len(train_loader) == 0:
        raise RuntimeError('Train loader is empty. Check dataset_root and category.')
    train_loader_infinite = InfiniteDataloader(train_loader)
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

    resolved_train_steps = resolve_train_steps(config, train_loader)
    eval_interval_steps = resolve_eval_interval(config, len(train_loader))
    effective_epochs = resolved_train_steps / max(1, len(train_loader))
    dataset_paths['effective_epochs'] = effective_epochs
    runtime_summary = print_runtime_summary(
        config,
        paths,
        dataset_paths,
        resolved_train_steps,
        device,
        len(train_loader),
    )
    log_run_message(
        paths,
        'Runtime summary:\n'
        + json.dumps(runtime_summary, indent=2, ensure_ascii=False),
        print_message=False,
    )

    pretrain_penalty = str(config.imagenet_dir).lower() != 'none'
    if pretrain_penalty:
        penalty_transform_ops = [
            transforms.Resize((2 * config.image_size, 2 * config.image_size)),
            transforms.RandomGrayscale(0.3),
        ]
        if config.center_crop:
            penalty_transform_ops.append(transforms.CenterCrop(config.center_crop))
        else:
            penalty_transform_ops.append(transforms.CenterCrop(config.image_size))
        penalty_transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalization['mean'],
                std=config.normalization['std'],
            ),
        ])
        penalty_transform = transforms.Compose(penalty_transform_ops)
        penalty_set = ImageFolderWithoutTarget(
            config.imagenet_dir,
            transform=penalty_transform,
        )
        penalty_loader = DataLoader(
            penalty_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    teacher, student, autoencoder, evidence_fuser = build_models(config, device)
    teacher.eval()
    student.train()
    autoencoder.train()
    if evidence_fuser is not None:
        evidence_fuser.eval()

    teacher_mean, teacher_std = teacher_normalization(
        teacher=teacher,
        train_loader=train_loader,
        device=device,
    )

    accumulation_steps = max(1, int(config.accumulate_grad_batches))
    optimizer_parameter_groups = [student.parameters(), autoencoder.parameters()]
    optimizer = torch.optim.Adam(
        itertools.chain(*optimizer_parameter_groups),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps = max(1, math.ceil(resolved_train_steps / accumulation_steps))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(0.95 * optimizer_steps)),
        gamma=0.1,
    )
    use_amp = config.precision in {16, '16', '16-mixed'} and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    metric_logger = create_metric_logger(config, paths)
    tensorboard_writer = create_tensorboard_writer(config, paths)
    best_metric_value = None
    best_step = None
    latest_auc = None
    final_q_values = None

    tqdm_obj = tqdm(
        range(1, resolved_train_steps + 1),
        desc='Training EfficientAD',
    )
    for iteration in tqdm_obj:
        (image_st, image_ae) = next(train_loader_infinite)
        image_penalty = next(penalty_loader_infinite)

        image_st = image_st.to(device, non_blocking=pin_memory)
        image_ae = image_ae.to(device, non_blocking=pin_memory)
        if image_penalty is not None:
            image_penalty = image_penalty.to(device, non_blocking=pin_memory)

        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

            student_output_st = student(image_st)[:, :config.teacher_out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            loss_hard = hard_feature_loss(distance_st)

            if image_penalty is not None:
                student_output_penalty = student(image_penalty)[
                    :,
                    :config.teacher_out_channels,
                ]
                loss_penalty = torch.mean(student_output_penalty ** 2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_penalty = torch.tensor(0.0, device=device)
                loss_st = loss_hard

            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

            student_output_ae_full = student(image_ae)
            student_output_ae = student_output_ae_full[:, config.teacher_out_channels:]
            distance_ae = (teacher_output_ae - ae_output) ** 2
            distance_stae = (ae_output - student_output_ae) ** 2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae
            backward_loss = loss_total / accumulation_steps

        if use_amp:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        should_step = (
            iteration % accumulation_steps == 0
            or iteration == resolved_train_steps
        )
        if should_step:
            if config.gradient_clip_val and config.gradient_clip_val > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                clip_parameter_groups = [student.parameters(), autoencoder.parameters()]
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(*clip_parameter_groups),
                    config.gradient_clip_val,
                )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if iteration % config.log_every_n_steps == 0 or iteration == 1:
            train_metrics = {
                'loss_total': float(loss_total.item()),
                'loss_st': float(loss_st.item()),
                'loss_ae': float(loss_ae.item()),
                'loss_stae': float(loss_stae.item()),
                'loss_penalty': float(loss_penalty.item()),
                'lr': float(optimizer.param_groups[0]['lr']),
            }
            log_metrics(
                metric_logger,
                tensorboard_writer,
                'train',
                iteration,
                train_metrics,
            )
            tqdm_obj.set_description(
                f'Train loss {loss_total.item():.4f} | step {iteration}/{resolved_train_steps}'
            )

        if config.enable_checkpointing and iteration % config.save_every_n_steps == 0:
            try:
                save_checkpoint_bundle(
                    Path(paths['step_checkpoints_dir']) / f'step_{iteration:07d}',
                    teacher,
                    student,
                    autoencoder,
                    {
                        'step': iteration,
                        'type': 'step',
                        'model_variant': config.variant,
                    },
                    extra_modules={'evidence_fuser': evidence_fuser},
                )
            except (OSError, RuntimeError) as exc:
                log_run_message(
                    paths,
                    (
                        f'Warning: step checkpoint at step {iteration} could not be saved '
                        f'and was skipped: {exc}'
                    ),
                )

        if eval_interval_steps and iteration % eval_interval_steps == 0 and iteration < resolved_train_steps:
            teacher.eval()
            student.eval()
            autoencoder.eval()
            if evidence_fuser is not None:
                evidence_fuser.eval()
            evaluation = evaluate_and_maybe_checkpoint(
                step=iteration,
                config=config,
                device=device,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                evidence_fuser=evidence_fuser,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                validation_loader=validation_loader,
                test_loader=test_loader,
                dataset_paths=dataset_paths,
                paths=paths,
                default_transform=base_transform,
                log_fn=metric_logger,
                writer=tensorboard_writer,
                best_metric_value=best_metric_value,
                best_step=best_step,
                desc_prefix='Intermediate',
            )
            latest_auc = evaluation['auc']
            best_metric_value = evaluation['best_metric_value']
            best_step = evaluation['best_step']
            final_q_values = {
                'q_st_start': float(evaluation['q_st_start']),
                'q_st_end': float(evaluation['q_st_end']),
                'q_ae_start': float(evaluation['q_ae_start']),
                'q_ae_end': float(evaluation['q_ae_end']),
            }
            log_run_message(paths, f'Intermediate image auc: {latest_auc:.4f}')
            teacher.eval()
            student.train()
            autoencoder.train()
            if evidence_fuser is not None:
                evidence_fuser.eval()

    teacher.eval()
    student.eval()
    autoencoder.eval()
    if evidence_fuser is not None:
        evidence_fuser.eval()

    save_torch_object_atomic(teacher, Path(paths['train_output_dir']) / 'teacher_final.pth')
    save_torch_object_atomic(student, Path(paths['train_output_dir']) / 'student_final.pth')
    save_torch_object_atomic(
        autoencoder,
        Path(paths['train_output_dir']) / 'autoencoder_final.pth',
    )
    if evidence_fuser is not None:
        save_torch_object_atomic(
            evidence_fuser,
            Path(paths['train_output_dir']) / 'evidence_fuser_final.pth',
        )

    if config.enable_checkpointing and config.save_last:
        save_checkpoint_bundle(
            paths['last_checkpoint_dir'],
            teacher,
            student,
            autoencoder,
            {
                'step': resolved_train_steps,
                'type': 'last',
                'model_variant': config.variant,
            },
            extra_modules={'evidence_fuser': evidence_fuser},
        )

    final_auc = None
    if config.run_test_after_fit:
        evaluation = evaluate_and_maybe_checkpoint(
            step=resolved_train_steps,
            config=config,
            device=device,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            evidence_fuser=evidence_fuser,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            validation_loader=validation_loader,
            test_loader=test_loader,
            dataset_paths=dataset_paths,
            paths=paths,
            default_transform=base_transform,
            log_fn=metric_logger,
            writer=tensorboard_writer,
            best_metric_value=best_metric_value,
            best_step=best_step,
            desc_prefix='Final',
        )
        final_auc = evaluation['auc']
        best_metric_value = evaluation['best_metric_value']
        best_step = evaluation['best_step']
        final_q_values = {
            'q_st_start': float(evaluation['q_st_start']),
            'q_st_end': float(evaluation['q_st_end']),
            'q_ae_start': float(evaluation['q_ae_start']),
            'q_ae_end': float(evaluation['q_ae_end']),
        }
        log_run_message(paths, f'Final image auc: {final_auc:.4f}')

    run_summary = {
        'config_path': config.config,
        'project_name': config.project_name,
        'experiment_name': config.experiment_name,
        'dataset_name': config.dataset_name,
        'category': config.category,
        'model_name': config.model_name,
        'model_variant': config.variant,
        'model_size': config.model_size,
        'seed': config.seed,
        'train_steps': resolved_train_steps,
        'max_epochs': config.max_epochs,
        'effective_epochs': effective_epochs,
        'image_size': config.image_size,
        'batch_size': config.batch_size,
        'val_batch_size': config.val_batch_size,
        'test_batch_size': config.test_batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'teacher_out_channels': config.teacher_out_channels,
        'padding': config.padding,
        'pad_maps': config.pad_maps,
        'device': str(device),
        'train_dir_name': config.train_dir_name,
        'val_dir_name': config.val_dir_name,
        'test_dir_name': config.test_dir_name,
        'eval_split': test_split,
        'data_path': category_root,
        'train_path': train_path,
        'validation_path': (
            validation_path
            if validation_path is not None
            else f'{train_path} (random 10% split)'
        ),
        'test_path': test_path,
        'split_csv_file': getattr(config, 'split_csv_file', None),
        'split_csv_path': dataset_paths.get('split_csv_path'),
        'pretrained_teacher': config.pretrained_teacher,
        'imagenet_dir': config.imagenet_dir,
        'output_dir': paths['output_dir'],
        'train_output_dir': paths['train_output_dir'],
        'resolved_config_path': paths['resolved_config_path'],
        'run_log_path': paths['run_log_path'],
        'test_output_dir': (
            paths['test_output_dir'] if config.save_anomaly_maps else None
        ),
        'predictions_csv_path': (
            paths['predictions_csv_path'] if config.save_predictions else None
        ),
        'metrics_csv_path': (
            paths['metrics_csv_path'] if config.enable_csv_logger else None
        ),
        'best_checkpoint_dir': (
            paths['best_checkpoint_dir']
            if config.enable_checkpointing and config.save_top_k
            else None
        ),
        'last_checkpoint_dir': (
            paths['last_checkpoint_dir']
            if config.enable_checkpointing and config.save_last
            else None
        ),
        'best_image_auc': best_metric_value,
        'best_step': best_step,
        'final_image_auc': final_auc,
        'latest_intermediate_auc': latest_auc,
        'q_values': final_q_values,
    }
    with open(paths['run_summary_path'], 'w', encoding='utf-8') as file:
        json.dump(run_summary, file, indent=2)
    log_run_message(
        paths,
        'Run summary:\n' + json.dumps(run_summary, indent=2, ensure_ascii=False),
        print_message=False,
    )

    if tensorboard_writer is not None:
        tensorboard_writer.close()

    return run_summary


def test(
    test_loader,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    teacher_mean,
    teacher_std,
    out_channels,
    device,
    default_transform,
    q_st_start,
    q_st_end,
    q_ae_start,
    q_ae_end,
    test_output_dir=None,
    predictions_csv_path=None,
    visualizations_dir=None,
    save_anomaly_maps=True,
    save_visualizations=False,
    pad_maps=True,
    threshold=None,
    dataset_name='mvtec_ad',
    desc='Running inference',
):
    y_true = []
    y_score = []
    record_paths = []
    anomaly_map_save_disabled = False

    for images, _, paths in tqdm(test_loader, desc=desc):
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
            orig_width, orig_height = original_sizes[index]
            current_map = map_combined[index:index + 1]
            if pad_maps:
                current_map = F.pad(current_map, (4, 4, 4, 4))
            current_map = F.interpolate(
                current_map,
                (orig_height, orig_width),
                mode='bilinear',
            )
            current_map = current_map[0, 0].detach().cpu().numpy()
            current_map = np.nan_to_num(
                current_map,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            defect_class = infer_defect_class_from_path(path)
            img_name = Path(path).stem
            if save_anomaly_maps and test_output_dir is not None and not anomaly_map_save_disabled:
                target_dir = Path(test_output_dir) / defect_class
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / f'{img_name}.tiff'
                try:
                    tifffile.imwrite(
                        target_path,
                        np.asarray(current_map, dtype=np.float32),
                    )
                except OSError as error:
                    anomaly_map_save_disabled = True
                    if target_path.exists() and target_path.stat().st_size == 0:
                        try:
                            target_path.unlink()
                        except OSError:
                            pass
                    print(
                        'Warning: failed to save anomaly maps to '
                        f'{target_dir}. Disabling anomaly map saving for the '
                        f'remainder of this run. Original error: {error}'
                    )
            if save_visualizations and visualizations_dir is not None:
                target_dir = Path(visualizations_dir) / defect_class
                target_dir.mkdir(parents=True, exist_ok=True)
                save_visualization_image(current_map, target_dir / f'{img_name}.png')

            y_true_image = infer_image_label_from_path(path)
            y_score_image = float(np.max(current_map))
            y_true.append(y_true_image)
            y_score.append(y_score_image)
            record_paths.append(path)

    prediction_rows = build_prediction_rows(y_true, y_score, record_paths, threshold)
    if predictions_csv_path is not None:
        write_predictions_csv(predictions_csv_path, prediction_rows)

    if roc_auc_score is None:
        raise RuntimeError(
            'scikit-learn is required for AUROC metrics. Please install it with: '
            'pip install scikit-learn'
        )
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100, prediction_rows


@torch.no_grad()
def predict(
    image,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    teacher_mean,
    teacher_std,
    out_channels,
    q_st_start=None,
    q_st_end=None,
    q_ae_start=None,
    q_ae_end=None,
    return_intermediates=False,
):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean(
        (teacher_output - student_output[:, :out_channels]) ** 2,
        dim=1,
        keepdim=True,
    )
    map_ae = torch.mean(
        (autoencoder_output - student_output[:, out_channels:]) ** 2,
        dim=1,
        keepdim=True,
    )
    map_st = normalize_anomaly_map(map_st, q_st_start, q_st_end)
    map_ae = normalize_anomaly_map(map_ae, q_ae_start, q_ae_end)
    if evidence_fuser is not None:
        map_combined = evidence_fuser(map_st, map_ae)
    else:
        map_combined = 0.5 * map_st + 0.5 * map_ae
    map_combined = torch.nan_to_num(map_combined, nan=0.0, posinf=0.0, neginf=0.0)
    if return_intermediates:
        return map_combined, map_st, map_ae, {
            'teacher_output': teacher_output,
            'student_output': student_output,
            'autoencoder_output': autoencoder_output,
            'local_anomaly_map': map_st,
            'global_anomaly_map': map_ae,
            'final_anomaly_map': map_combined,
        }
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(
    validation_loader,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    out_channels,
    device,
    desc='Map normalization',
):
    maps_st = []
    maps_ae = []
    for image, _ in tqdm(validation_loader, desc=desc):
        image = image.to(device)
        _, map_st, map_ae = predict(
            image=image,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            evidence_fuser=None,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            out_channels=out_channels,
        )
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader, device):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


def require_sklearn_metrics():
    if roc_auc_score is None or precision_recall_curve is None:
        raise RuntimeError(
            'scikit-learn is required for CFRP metrics. Please install it with: '
            'pip install scikit-learn'
        )


def format_float_metric(value):
    if value is None or not np.isfinite(value):
        return 'nan'
    return f'{value:.6f}'


def safe_roc_auc(y_true, y_score, metric_name):
    require_sklearn_metrics()
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if np.unique(y_true).size < 2:
        print(f'{metric_name} is undefined because only one class is present.')
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def f1_max_score(y_true, y_score):
    require_sklearn_metrics()
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if np.unique(y_true).size < 2:
        print('F1-max is undefined because only one pixel class is present.')
        return float('nan')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    denominator = precision + recall
    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.max(f1_values))


def cfrp_split_path():
    return Path(__file__).resolve().parent / DEFAULT_CFRP_SPLIT_PATH


def state_dict_to_cpu(module):
    if module is None:
        return None
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


def save_cfrp_final_model(
    model_path,
    config,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    teacher_mean,
    teacher_std,
    q_values,
):
    payload = {
        'format': 'efficientad_cfrp_min_v1',
        'dataset': 'cfrp',
        'image_size': int(config.image_size),
        'model_size': config.model_size,
        'model_variant': config.variant,
        'teacher_out_channels': int(config.teacher_out_channels),
        'padding': bool(config.padding),
        'pad_maps': bool(config.pad_maps),
        'hf_fusion_beta': float(config.hf_fusion_beta),
        'normalization': config.normalization,
        'teacher': state_dict_to_cpu(teacher),
        'student': state_dict_to_cpu(student),
        'autoencoder': state_dict_to_cpu(autoencoder),
        'evidence_fuser': state_dict_to_cpu(evidence_fuser),
        'teacher_mean': teacher_mean.detach().cpu(),
        'teacher_std': teacher_std.detach().cpu(),
        'q_values': {
            name: value.detach().cpu()
            for name, value in q_values.items()
        },
    }
    save_torch_object_atomic(payload, model_path)


def apply_cfrp_checkpoint_config(config, checkpoint):
    for attr in (
        'model_size',
        'teacher_out_channels',
        'padding',
        'pad_maps',
        'hf_fusion_beta',
        'normalization',
    ):
        if attr in checkpoint:
            setattr(config, attr, checkpoint[attr])
    variant = checkpoint.get('model_variant') or checkpoint.get('variant')
    if variant is not None:
        setattr(config, 'variant', variant)
        setattr(config, 'model_variant', variant)


def load_cfrp_model_state(
    checkpoint,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    device,
):
    if checkpoint.get('teacher') is not None:
        teacher.load_state_dict(checkpoint['teacher'])
    student.load_state_dict(checkpoint['student'])
    autoencoder.load_state_dict(checkpoint['autoencoder'])
    if checkpoint.get('evidence_fuser') is not None:
        if evidence_fuser is None:
            raise RuntimeError(
                'The checkpoint contains an evidence_fuser, but the selected '
                'model variant did not create one.'
            )
        evidence_fuser.load_state_dict(checkpoint['evidence_fuser'])

    teacher.to(device).eval()
    student.to(device).eval()
    autoencoder.to(device).eval()
    if evidence_fuser is not None:
        evidence_fuser.to(device).eval()


@torch.no_grad()
def cfrp_map_normalization(
    validation_loader,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    out_channels,
    device,
    default_transform,
):
    maps_st = []
    maps_ae = []
    for images, _, _, _, _ in tqdm(validation_loader, desc='CFRP map normalization'):
        batch = torch.stack([default_transform(image) for image in images], dim=0)
        batch = batch.to(device)
        _, map_st, map_ae = predict(
            image=batch,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            evidence_fuser=None,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            out_channels=out_channels,
        )
        maps_st.append(map_st.detach().cpu())
        maps_ae.append(map_ae.detach().cpu())

    if not maps_st:
        raise RuntimeError('CFRP validation loader is empty; cannot normalize maps.')

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    return {
        'q_st_start': torch.quantile(maps_st, q=0.9),
        'q_st_end': torch.quantile(maps_st, q=0.995),
        'q_ae_start': torch.quantile(maps_ae, q=0.9),
        'q_ae_end': torch.quantile(maps_ae, q=0.995),
    }


def normalize_array_01(array):
    array = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - minimum) / (maximum - minimum), 0.0, 1.0)


def cfrp_heatmap(norm_map):
    heatmap = np.zeros((*norm_map.shape, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(2.0 * norm_map, 0.0, 1.0)
    heatmap[..., 1] = np.clip(2.0 * norm_map - 0.5, 0.0, 1.0)
    heatmap[..., 2] = np.clip(1.0 - 2.0 * norm_map, 0.0, 1.0) * 0.35
    return np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)


def save_cfrp_visualization(image, mask, anomaly_map, file_path, image_size):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    panel_edge = min(int(image_size), 300)
    panel_size = (panel_edge, panel_edge)

    input_image = image.resize(panel_size, Image.BILINEAR)
    input_np = np.asarray(input_image.convert('RGB'), dtype=np.uint8)

    mask_image = mask.resize(panel_size, Image.NEAREST)
    mask_np = np.asarray(mask_image, dtype=np.uint8)
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)

    norm_map = normalize_array_01(anomaly_map)
    map_image = Image.fromarray(np.uint8(norm_map * 255), mode='L')
    map_image = map_image.resize(panel_size, Image.BILINEAR)
    map_np = np.asarray(map_image, dtype=np.float32) / 255.0
    heatmap_np = cfrp_heatmap(map_np)
    overlay = np.uint8(0.65 * input_np.astype(np.float32) + 0.35 * heatmap_np)

    canvas = np.concatenate([input_np, mask_rgb, heatmap_np, overlay], axis=1)
    Image.fromarray(canvas, mode='RGB').save(
        file_path,
        quality=85,
        optimize=True,
    )


def write_cfrp_train_log(log_path, config, final_loss, model_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        'dataset: cfrp',
        f'image_size: {config.image_size}',
        f'batch_size: {config.batch_size}',
        f'steps: {config.train_steps}',
        f'final loss: {format_float_metric(final_loss)}',
        f'model path: {Path(model_path)}',
    ]
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_cfrp_training(config):
    config = apply_runtime_defaults(config)
    configure_runtime(config.seed, config.deterministic)
    device = resolve_device(config)
    pin_memory = device.type == 'cuda'

    save_dir = Path(config.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    split_path = cfrp_split_path()
    split, image_dir, mask_dir = load_or_create_cfrp_split(
        config.dataset_root,
        split_path=split_path,
    )
    if not split['train']:
        raise RuntimeError('CFRP train split is empty.')
    normalization_split = 'val' if split['val'] else 'test'
    if not split[normalization_split]:
        raise RuntimeError('CFRP val/test split is empty; cannot normalize maps.')

    base_transform, train_transform = build_transforms(config)
    train_set = CfrpTrainDataset(
        config.dataset_root,
        split='train',
        transform=train_transform,
        split_path=split_path,
    )
    normalization_set = CfrpEvalDataset(
        config.dataset_root,
        split=normalization_split,
        split_path=split_path,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    normalization_loader = DataLoader(
        normalization_set,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cfrp_eval_batch,
    )
    if len(train_loader) == 0:
        raise RuntimeError('CFRP train loader is empty.')
    train_loader_infinite = InfiniteDataloader(train_loader)

    print('CFRP runtime summary:')
    print(json.dumps({
        'dataset': 'cfrp',
        'image_dir': str(image_dir),
        'mask_dir': str(mask_dir),
        'split_file': str(split_path),
        'train': len(split['train']),
        'val': len(split['val']),
        'test': len(split['test']),
        'normalization_split': normalization_split,
        'image_size': config.image_size,
        'batch_size': config.batch_size,
        'steps': config.train_steps,
        'save_dir': str(save_dir),
        'device': str(device),
    }, indent=2, ensure_ascii=False))

    teacher, student, autoencoder, evidence_fuser = build_models(config, device)
    teacher.eval()
    student.train()
    autoencoder.train()
    if evidence_fuser is not None:
        evidence_fuser.eval()

    teacher_mean, teacher_std = teacher_normalization(
        teacher=teacher,
        train_loader=train_loader,
        device=device,
    )

    accumulation_steps = max(1, int(config.accumulate_grad_batches))
    optimizer = torch.optim.Adam(
        itertools.chain(student.parameters(), autoencoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps = max(1, math.ceil(config.train_steps / accumulation_steps))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(0.95 * optimizer_steps)),
        gamma=0.1,
    )
    use_amp = config.precision in {16, '16', '16-mixed'} and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    final_loss = float('nan')
    tqdm_obj = tqdm(range(1, int(config.train_steps) + 1), desc='Training CFRP EfficientAD')
    for iteration in tqdm_obj:
        image_st, image_ae = next(train_loader_infinite)
        image_st = image_st.to(device, non_blocking=pin_memory)
        image_ae = image_ae.to(device, non_blocking=pin_memory)

        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

            student_output_st = student(image_st)[:, :config.teacher_out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            loss_hard = hard_feature_loss(distance_st)
            loss_st = loss_hard

            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

            student_output_ae_full = student(image_ae)
            student_output_ae = student_output_ae_full[:, config.teacher_out_channels:]
            loss_ae = torch.mean((teacher_output_ae - ae_output) ** 2)
            loss_stae = torch.mean((ae_output - student_output_ae) ** 2)
            loss_total = loss_st + loss_ae + loss_stae
            backward_loss = loss_total / accumulation_steps

        if use_amp:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        should_step = (
            iteration % accumulation_steps == 0
            or iteration == int(config.train_steps)
        )
        if should_step:
            if config.gradient_clip_val and config.gradient_clip_val > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(student.parameters(), autoencoder.parameters()),
                    config.gradient_clip_val,
                )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        final_loss = float(loss_total.item())
        if iteration % config.log_every_n_steps == 0 or iteration == 1:
            tqdm_obj.set_description(
                f'CFRP loss {final_loss:.4f} | step {iteration}/{config.train_steps}'
            )

    teacher.eval()
    student.eval()
    autoencoder.eval()
    if evidence_fuser is not None:
        evidence_fuser.eval()

    q_values = cfrp_map_normalization(
        validation_loader=normalization_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        out_channels=config.teacher_out_channels,
        device=device,
        default_transform=base_transform,
    )

    model_path = save_dir / 'final_model.pth'
    save_cfrp_final_model(
        model_path=model_path,
        config=config,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        evidence_fuser=evidence_fuser,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        q_values=q_values,
    )
    write_cfrp_train_log(save_dir / 'train_log.txt', config, final_loss, model_path)
    print(f'CFRP final model saved to: {model_path}')
    print(f'CFRP train log saved to: {save_dir / "train_log.txt"}')
    return {
        'model_path': str(model_path),
        'train_log_path': str(save_dir / 'train_log.txt'),
        'final_loss': final_loss,
    }


@torch.no_grad()
def evaluate_cfrp_test(
    test_loader,
    teacher,
    student,
    autoencoder,
    evidence_fuser,
    teacher_mean,
    teacher_std,
    q_values,
    out_channels,
    device,
    default_transform,
    save_dir,
    save_vis,
    image_size,
    pad_maps=True,
):
    image_labels = []
    image_scores = []
    pixel_labels = []
    pixel_scores = []
    score_rows = []
    saved_visualizations = 0
    vis_dir = Path(save_dir) / 'vis'
    save_limit = max(0, int(save_vis))
    if save_limit > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
        for old_file in vis_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()

    q_st_start = q_values['q_st_start'].to(device)
    q_st_end = q_values['q_st_end'].to(device)
    q_ae_start = q_values['q_ae_start'].to(device)
    q_ae_end = q_values['q_ae_end'].to(device)

    print('Image-level anomaly scores:')
    for images, masks, labels, paths, filenames in tqdm(test_loader, desc='Testing CFRP'):
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

        for index, filename in enumerate(filenames):
            mask = masks[index]
            target_width, target_height = mask.size
            current_map = map_combined[index:index + 1]
            if pad_maps:
                current_map = F.pad(current_map, (4, 4, 4, 4))
            current_map = F.interpolate(
                current_map,
                (target_height, target_width),
                mode='bilinear',
            )
            current_map = current_map[0, 0].detach().cpu().numpy()
            current_map = np.nan_to_num(
                current_map,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
            image_label = int(labels[index].item())
            mask_sum = int(mask_np.sum())
            image_score = float(np.max(current_map))
            image_labels.append(image_label)
            image_scores.append(image_score)
            pixel_labels.append(mask_np.reshape(-1))
            pixel_scores.append(current_map.reshape(-1))
            score_rows.append({
                'filename': filename,
                'label': image_label,
                'mask_sum': mask_sum,
                'image_score': image_score,
            })
            print(f'{filename}\tlabel={image_label}\tscore={image_score:.6f}')

            if saved_visualizations < save_limit:
                output_path = vis_dir / f'{Path(filename).stem}_vis.jpg'
                save_cfrp_visualization(
                    images[index],
                    mask,
                    current_map,
                    output_path,
                    image_size=image_size,
                )
                saved_visualizations += 1

    if not image_labels:
        raise RuntimeError('CFRP test split is empty.')

    pixel_labels = np.concatenate(pixel_labels)
    pixel_scores = np.concatenate(pixel_scores)
    image_labels_array = np.asarray(image_labels).astype(np.uint8)
    image_auc_reason = None
    if np.unique(image_labels_array).size < 2:
        image_auc = float('nan')
        image_auc_reason = (
            'because only one image-level class exists in this split.'
        )
        print(f'Image AUROC: nan, {image_auc_reason}')
    else:
        image_auc = safe_roc_auc(image_labels, image_scores, 'Image AUROC')

    metrics = {
        'image_auroc': image_auc,
        'image_auroc_reason': image_auc_reason,
        'pixel_auroc': safe_roc_auc(pixel_labels, pixel_scores, 'Pixel AUROC'),
        'f1_max': f1_max_score(pixel_labels, pixel_scores),
        'num_test_images': len(image_labels),
        'num_positive_images': int(np.sum(image_labels)),
        'num_negative_images': int(len(image_labels) - np.sum(image_labels)),
        'num_positive_pixels': int(np.sum(pixel_labels)),
        'num_negative_pixels': int(pixel_labels.size - np.sum(pixel_labels)),
        'saved_visualizations': saved_visualizations,
    }
    return metrics, score_rows


def write_cfrp_metrics(metrics_path, metrics):
    image_auc_line = f"Image AUROC: {format_float_metric(metrics['image_auroc'])}"
    if metrics.get('image_auroc_reason'):
        image_auc_line += f", {metrics['image_auroc_reason']}"
    lines = [
        f"Eval split: {metrics.get('eval_split', 'unknown')}",
        image_auc_line,
        f"Pixel AUROC: {format_float_metric(metrics['pixel_auroc'])}",
        f"F1-max: {format_float_metric(metrics['f1_max'])}",
        f"Test images: {metrics['num_test_images']}",
        f"Positive images: {metrics['num_positive_images']}",
        f"Negative images: {metrics['num_negative_images']}",
        f"Positive pixels: {metrics['num_positive_pixels']}",
        f"Negative pixels: {metrics['num_negative_pixels']}",
        f"Saved visualizations: {metrics['saved_visualizations']}",
    ]
    Path(metrics_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_cfrp_scores_csv(scores_path, rows):
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['filename', 'label', 'mask_sum', 'image_score'],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'filename': row['filename'],
                'label': int(row['label']),
                'mask_sum': int(row['mask_sum']),
                'image_score': f"{float(row['image_score']):.8f}",
            })


def run_cfrp_test(config):
    config = apply_runtime_defaults(config)
    if not config.checkpoint:
        raise ValueError('--checkpoint is required when --mode test is used.')
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Could not find checkpoint: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    apply_cfrp_checkpoint_config(config, checkpoint)
    device = resolve_device(config)
    pin_memory = device.type == 'cuda'
    save_dir = Path(config.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_image_size = checkpoint.get('image_size')
    if saved_image_size is not None and int(saved_image_size) != int(config.image_size):
        print(
            'Warning: checkpoint image_size is '
            f'{saved_image_size}, but CLI image_size is {config.image_size}.'
        )

    base_transform, _ = build_transforms(config)
    split_path = cfrp_split_path()
    split, image_dir, mask_dir = load_or_create_cfrp_split(
        config.dataset_root,
        split_path=split_path,
    )
    eval_split = str(getattr(config, 'eval_split', 'test') or 'test').lower()
    if eval_split == 'test_public':
        eval_split = 'test'
    if eval_split not in {'test', 'val', 'all'}:
        raise ValueError('CFRP --eval_split must be one of: val, test, all.')
    if eval_split == 'all':
        print(
            'Warning: --eval_split all includes train samples. Use it as a '
            'baseline/debug view, not as the standard anomaly-detection test protocol.'
        )
    test_set = CfrpEvalDataset(
        config.dataset_root,
        split=eval_split,
        split_path=split_path,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cfrp_eval_batch,
    )

    teacher, student, autoencoder, evidence_fuser = build_models(config, device)
    load_cfrp_model_state(
        checkpoint,
        teacher,
        student,
        autoencoder,
        evidence_fuser,
        device,
    )
    q_values = checkpoint['q_values']
    teacher_mean = checkpoint['teacher_mean'].to(device)
    teacher_std = checkpoint['teacher_std'].to(device)

    print('CFRP test summary:')
    print(json.dumps({
        'dataset': 'cfrp',
        'image_dir': str(image_dir),
        'mask_dir': str(mask_dir),
        'split_file': str(split_path),
        'eval_split': eval_split,
        'eval_images': len(test_set),
        'checkpoint': str(checkpoint_path),
        'save_dir': str(save_dir),
        'save_vis': int(config.save_vis),
        'device': str(device),
    }, indent=2, ensure_ascii=False))

    metrics, score_rows = evaluate_cfrp_test(
        test_loader=test_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        evidence_fuser=evidence_fuser,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        q_values=q_values,
        out_channels=config.teacher_out_channels,
        device=device,
        default_transform=base_transform,
        save_dir=save_dir,
        save_vis=config.save_vis,
        image_size=config.image_size,
        pad_maps=config.pad_maps,
    )
    metrics['eval_split'] = eval_split

    metrics_path = save_dir / 'metrics.txt'
    scores_path = save_dir / 'scores.csv'
    write_cfrp_metrics(metrics_path, metrics)
    write_cfrp_scores_csv(scores_path, score_rows)
    image_auc_message = f"Image AUROC: {format_float_metric(metrics['image_auroc'])}"
    if metrics.get('image_auroc_reason'):
        image_auc_message += f", {metrics['image_auroc_reason']}"
    print(image_auc_message)
    print(f"Pixel AUROC: {format_float_metric(metrics['pixel_auroc'])}")
    print(f"F1-max: {format_float_metric(metrics['f1_max'])}")
    print(f'CFRP metrics saved to: {metrics_path}')
    print(f'CFRP scores saved to: {scores_path}')
    print(f'CFRP visualizations saved to: {save_dir / "vis"}')
    return metrics


def main():
    runtime_config = get_argparse()
    if getattr(runtime_config, 'dataset_name', getattr(runtime_config, 'dataset', None)) == 'cfrp':
        if runtime_config.mode == 'test':
            run_cfrp_test(runtime_config)
        else:
            run_cfrp_training(runtime_config)
        return
    if getattr(runtime_config, 'mode', 'train') != 'train':
        raise ValueError('--mode test is currently implemented for --dataset cfrp.')
    run_training(runtime_config)


if __name__ == '__main__':
    main()
