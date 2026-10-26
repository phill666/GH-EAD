import copy
import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from btad_dataset import DEFAULT_BTAD_ROOT
from efficientad_variant_gs_hf_v2 import BASELINE_VARIANT, normalize_model_variant
from visa_dataset import DEFAULT_VISA_SPLIT_CSV


DEFAULT_CONFIG = {
    'basic': {
        'experiment_name': 'efficientad_mvtecad2',
        'project_name': 'EfficientAD',
        'seed': 42,
        'output_root': 'output/yaml_runs',
        'save_dir': 'auto',
        'device': 'cuda:0',
        'num_workers': 4,
    },
    'dataset': {
        'dataset_name': 'mvtec_ad',
        'dataset_root': 'data/mvtec_ad2',
        'category': 'vial',
        'split_csv_file': DEFAULT_VISA_SPLIT_CSV,
        'train_dir_name': 'train',
        'val_dir_name': 'validation',
        'test_dir_name': 'test',
        'use_validation': True,
        'use_test_public': True,
        'support_test_private': False,
        'image_size': 256,
        'center_crop': None,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'batch_size': 1,
        'val_batch_size': 4,
        'test_batch_size': 4,
    },
    'model': {
        'model_name': 'EfficientAD',
        'name': 'EfficientAD',
        'variant': BASELINE_VARIANT,
        'hf_fusion_beta': 0.2,
        'model_size': 's',
        'teacher_out_channels': 384,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'padding': False,
        'pad_maps': True,
        'imagenet_dir': 'none',
        'pretrained_teacher': 'models/teacher_small.pth',
        'use_pretrained_teacher': True,
    },
    'training': {
        'train_steps': 70000,
        'max_epochs': None,
        'accelerator': 'cuda',
        'precision': 32,
        'gradient_clip_val': 0.0,
        'accumulate_grad_batches': 1,
        'log_every_n_steps': 10,
        'check_val_every_n_epoch': None,
        'check_val_every_n_steps': 10000,
        'save_every_n_steps': 1000,
        'deterministic': False,
    },
    'logging': {
        'save_top_k': 1,
        'monitor_metric': 'image_auc',
        'monitor_mode': 'max',
        'save_last': True,
        'enable_checkpointing': True,
        'enable_tensorboard': False,
        'enable_csv_logger': True,
        'save_predictions': True,
        'save_visualizations': False,
    },
    'evaluation': {
        'run_test_after_fit': True,
        'eval_split': 'test_public',
        'metrics': ['image_auc'],
        'threshold': None,
        'save_anomaly_maps': True,
    },
}


def _strip_comments(line):
    in_single = False
    in_double = False
    result = []
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == '#' and not in_single and not in_double:
            break
        result.append(char)
    return ''.join(result).rstrip()


def _parse_scalar(value):
    value = value.strip()
    if value == '':
        return ''
    if value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]

    lowered = value.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    if lowered in {'null', 'none', '~'}:
        return None

    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = []
        current = []
        in_single = False
        in_double = False
        for char in inner:
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            if char == ',' and not in_single and not in_double:
                parts.append(''.join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            parts.append(''.join(current).strip())
        return [_parse_scalar(part) for part in parts]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def load_simple_yaml(path):
    text = Path(path).read_text(encoding='utf-8')
    raw_lines = text.splitlines()
    lines = []
    for raw_line in raw_lines:
        stripped = _strip_comments(raw_line)
        if stripped.strip():
            indent = len(stripped) - len(stripped.lstrip(' '))
            lines.append((indent, stripped.lstrip(' ')))

    if not lines:
        return {}

    root = {}
    stack = [(-1, root)]

    def next_content(index):
        for candidate_index in range(index + 1, len(lines)):
            return lines[candidate_index]
        return None

    for index, (indent, content) in enumerate(lines):
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith('- '):
            if not isinstance(parent, list):
                raise ValueError('Invalid YAML structure: list item without list parent.')
            parent.append(_parse_scalar(content[2:].strip()))
            continue

        key, separator, value = content.partition(':')
        if separator != ':':
            raise ValueError(f'Invalid YAML line: {content}')
        key = key.strip()
        value = value.strip()

        if value:
            if not isinstance(parent, dict):
                raise ValueError('Invalid YAML structure: key/value without dict parent.')
            parent[key] = _parse_scalar(value)
            continue

        upcoming = next_content(index)
        if upcoming is None or upcoming[0] <= indent:
            child = {}
        else:
            child = [] if upcoming[1].startswith('- ') else {}

        if not isinstance(parent, dict):
            raise ValueError('Invalid YAML structure: nested block without dict parent.')
        parent[key] = child
        stack.append((indent, child))

    return root


def load_config_file(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.json':
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    if suffix in {'.yaml', '.yml'}:
        return load_simple_yaml(path)
    raise ValueError(f'Unsupported config file format: {path}')


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_repo_path(path_value, repo_root):
    if path_value in (None, '', 'none'):
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((Path(repo_root) / path).resolve())


def normalize_model_size(model_size):
    normalized = str(model_size).strip().lower()
    if normalized in {'s', 'small'}:
        return 'small'
    if normalized in {'m', 'medium'}:
        return 'medium'
    raise ValueError(f'Unsupported model_size: {model_size}')


def validate_config(config):
    required_sections = ['basic', 'dataset', 'model', 'training', 'logging', 'evaluation']
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f'Missing or invalid config section: {section}')

    dataset = config['dataset']
    model = config['model']
    training = config['training']
    logging_cfg = config['logging']
    evaluation = config['evaluation']

    if 'model_name' not in model and 'name' in model:
        model['model_name'] = model['name']
    if 'name' not in model and 'model_name' in model:
        model['name'] = model['model_name']

    if dataset['dataset_name'] not in {'mvtec_ad', 'mvtec_loco', 'visa', 'btad'}:
        raise ValueError(f"Unsupported dataset_name: {dataset['dataset_name']}")
    if not dataset['category']:
        raise ValueError('dataset.category must not be empty.')
    if not dataset['dataset_root']:
        raise ValueError('dataset.dataset_root must not be empty.')
    if dataset['dataset_name'] == 'visa' and not dataset.get('split_csv_file'):
        raise ValueError('dataset.split_csv_file must not be empty for VisA configs.')
    if not isinstance(dataset['normalization'], dict):
        raise ValueError('dataset.normalization must be a mapping.')
    if len(dataset['normalization'].get('mean', [])) != 3:
        raise ValueError('dataset.normalization.mean must contain 3 values.')
    if len(dataset['normalization'].get('std', [])) != 3:
        raise ValueError('dataset.normalization.std must contain 3 values.')

    config['model']['model_size'] = normalize_model_size(model['model_size'])
    config['model']['variant'] = normalize_model_variant(
        model.get('variant', BASELINE_VARIANT)
    )
    if not 0.0 <= float(model.get('hf_fusion_beta', 0.2)) <= 1.0:
        raise ValueError('model.hf_fusion_beta must be within [0, 1].')
    if model['teacher_out_channels'] <= 0:
        raise ValueError('model.teacher_out_channels must be positive.')

    if training['train_steps'] in (None, 0) and training['max_epochs'] in (None, 0):
        raise ValueError('Either training.train_steps or training.max_epochs must be set.')
    if training['accumulate_grad_batches'] <= 0:
        raise ValueError('training.accumulate_grad_batches must be positive.')
    if training['log_every_n_steps'] <= 0:
        raise ValueError('training.log_every_n_steps must be positive.')
    if training['save_every_n_steps'] <= 0:
        raise ValueError('training.save_every_n_steps must be positive.')

    if logging_cfg['monitor_mode'] not in {'min', 'max'}:
        raise ValueError('logging.monitor_mode must be "min" or "max".')
    if logging_cfg['save_top_k'] not in {0, 1}:
        raise ValueError('logging.save_top_k currently supports only 0 or 1.')
    if logging_cfg['monitor_metric'] != 'image_auc':
        raise ValueError('logging.monitor_metric currently supports only "image_auc".')

    if not isinstance(evaluation['metrics'], list) or not evaluation['metrics']:
        raise ValueError('evaluation.metrics must be a non-empty list.')
    unsupported_metrics = [metric for metric in evaluation['metrics']
                           if metric != 'image_auc']
    if unsupported_metrics:
        raise ValueError(
            'Unsupported evaluation.metrics: ' + ', '.join(unsupported_metrics)
        )
    if (not dataset['support_test_private']
            and evaluation['eval_split'] in {'test_private', 'test_private_mixed'}):
        raise ValueError(
            'evaluation.eval_split points to a private split while '
            'dataset.support_test_private is false.'
        )


def compute_output_dir(config, repo_root):
    basic = config['basic']
    dataset = config['dataset']
    model = config['model']

    output_root = resolve_repo_path(basic['output_root'], repo_root)
    save_dir = basic.get('save_dir')
    if save_dir in (None, '', 'auto'):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        save_dir = os.path.join(
            str(model['model_name']).strip().lower(),
            normalize_model_variant(model.get('variant', BASELINE_VARIANT)),
            dataset['category'],
            f"{timestamp}_{basic['experiment_name']}_{model['model_size']}_seed{basic['seed']}",
        )
        return os.path.join(output_root, save_dir)
    if Path(save_dir).is_absolute():
        return str(Path(save_dir))
    return os.path.join(output_root, save_dir)


def build_runtime_namespace(config, config_path, repo_root):
    validate_config(config)

    basic = config['basic']
    dataset = config['dataset']
    model = config['model']
    training = config['training']
    logging_cfg = config['logging']
    evaluation = config['evaluation']

    runtime = {
        'config': str(Path(config_path).resolve()),
        'experiment_name': basic['experiment_name'],
        'project_name': basic['project_name'],
        'seed': basic['seed'],
        'output_root': resolve_repo_path(basic['output_root'], repo_root),
        'save_dir': basic['save_dir'],
        'output_dir': compute_output_dir(config, repo_root),
        'device': basic['device'],
        'num_workers': basic['num_workers'],
        'dataset_name': dataset['dataset_name'],
        'dataset_root': resolve_repo_path(dataset['dataset_root'], repo_root),
        'btad_path': resolve_repo_path(dataset.get('dataset_root', DEFAULT_BTAD_ROOT), repo_root),
        'category': dataset['category'],
        'split_csv_file': dataset.get('split_csv_file', DEFAULT_VISA_SPLIT_CSV),
        'train_dir_name': dataset['train_dir_name'],
        'val_dir_name': dataset['val_dir_name'],
        'test_dir_name': dataset['test_dir_name'],
        'use_validation': dataset['use_validation'],
        'use_test_public': dataset['use_test_public'],
        'support_test_private': dataset['support_test_private'],
        'image_size': dataset['image_size'],
        'center_crop': dataset['center_crop'],
        'normalization': dataset['normalization'],
        'batch_size': dataset['batch_size'],
        'val_batch_size': dataset['val_batch_size'],
        'test_batch_size': dataset['test_batch_size'],
        'model_name': model['model_name'],
        'model_variant': model['variant'],
        'variant': model['variant'],
        'hf_fusion_beta': float(model.get('hf_fusion_beta', 0.2)),
        'model_size': model['model_size'],
        'teacher_out_channels': model['teacher_out_channels'],
        'learning_rate': model['lr'],
        'weight_decay': model['weight_decay'],
        'padding': model['padding'],
        'pad_maps': model['pad_maps'],
        'imagenet_dir': resolve_repo_path(model['imagenet_dir'], repo_root),
        'pretrained_teacher': resolve_repo_path(model['pretrained_teacher'], repo_root),
        'use_pretrained_teacher': model['use_pretrained_teacher'],
        'train_steps': training['train_steps'],
        'max_epochs': training['max_epochs'],
        'accelerator': training['accelerator'],
        'precision': training['precision'],
        'gradient_clip_val': training['gradient_clip_val'],
        'accumulate_grad_batches': training['accumulate_grad_batches'],
        'log_every_n_steps': training['log_every_n_steps'],
        'check_val_every_n_epoch': training['check_val_every_n_epoch'],
        'check_val_every_n_steps': training['check_val_every_n_steps'],
        'save_every_n_steps': training['save_every_n_steps'],
        'deterministic': training['deterministic'],
        'save_top_k': logging_cfg['save_top_k'],
        'monitor_metric': logging_cfg['monitor_metric'],
        'monitor_mode': logging_cfg['monitor_mode'],
        'save_last': logging_cfg['save_last'],
        'enable_checkpointing': logging_cfg['enable_checkpointing'],
        'enable_tensorboard': logging_cfg['enable_tensorboard'],
        'enable_csv_logger': logging_cfg['enable_csv_logger'],
        'save_predictions': logging_cfg['save_predictions'],
        'save_visualizations': logging_cfg['save_visualizations'],
        'run_test_after_fit': evaluation['run_test_after_fit'],
        'eval_split': evaluation['eval_split'],
        'metrics': evaluation['metrics'],
        'threshold': evaluation['threshold'],
        'save_anomaly_maps': evaluation['save_anomaly_maps'],
        'raw_config': config,
    }
    return Namespace(**runtime)


def load_training_config(config_path, repo_root, overrides=None):
    config_data = load_config_file(config_path)
    merged = deep_merge(DEFAULT_CONFIG, config_data)
    if overrides:
        merged = deep_merge(merged, overrides)
    return build_runtime_namespace(merged, config_path, repo_root)


def format_config_summary(runtime_config):
    summary = {
        'config_path': runtime_config.config,
        'output_dir': runtime_config.output_dir,
        'experiment_name': runtime_config.experiment_name,
        'project_name': runtime_config.project_name,
        'seed': runtime_config.seed,
        'device': runtime_config.device,
        'dataset_name': runtime_config.dataset_name,
        'dataset_root': runtime_config.dataset_root,
        'category': runtime_config.category,
        'split_csv_file': getattr(runtime_config, 'split_csv_file', None),
        'model_name': runtime_config.model_name,
        'model_variant': runtime_config.model_variant,
        'hf_fusion_beta': runtime_config.hf_fusion_beta,
        'model_size': runtime_config.model_size,
        'teacher_out_channels': runtime_config.teacher_out_channels,
        'train_steps': runtime_config.train_steps,
        'max_epochs': runtime_config.max_epochs,
        'batch_size': runtime_config.batch_size,
        'val_batch_size': runtime_config.val_batch_size,
        'test_batch_size': runtime_config.test_batch_size,
        'image_size': runtime_config.image_size,
        'learning_rate': runtime_config.learning_rate,
        'weight_decay': runtime_config.weight_decay,
        'padding': runtime_config.padding,
        'pad_maps': runtime_config.pad_maps,
        'eval_split': runtime_config.eval_split,
        'metrics': runtime_config.metrics,
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)
