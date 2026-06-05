import argparse
import csv
import glob
import os
import sys
from typing import Dict, List

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GPointNet scalar energies for HiT-ADV saved clean/adversarial pairs."
    )
    parser.add_argument('--pairs_dir', type=str, default='saved_hitadv_pairs',
                        help='directory containing batch_*.npz saved by eval.py --save_attack_pairs')
    parser.add_argument('--gpointnet_dir', type=str, default='../GPointNet',
                        help='path to the GPointNet project directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to a GPointNet Lightning .ckpt checkpoint')
    parser.add_argument('--output_dir', type=str, default='energy_results',
                        help='directory for CSV/NPZ energy outputs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='torch device, e.g. cuda:0 or cpu')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='energy inference batch size')
    parser.add_argument('--target_num_point', type=int, default=-1,
                        help='target point count before GPointNet; -1 keeps the saved pair point count')
    parser.add_argument('--no_patch_layernorm', action='store_true', default=False,
                        help='disable LayerNorm shape patching; input point count must match checkpoint config')
    parser.add_argument('--normalize', type=str, default='none',
                        choices=['none', 'provided_minmax', 'per_file_minmax'],
                        help='normalization before energy computation')
    parser.add_argument('--norm_min', type=float, default=None,
                        help='min value for --normalize provided_minmax')
    parser.add_argument('--norm_max', type=float, default=None,
                        help='max value for --normalize provided_minmax')
    parser.add_argument('--file_pattern', type=str, default='batch_*.npz',
                        help='glob pattern inside pairs_dir')
    parser.add_argument('--target_label', type=int, default=None,
                        help='filter samples to only this label before energy computation; skip files with no matches')
    return parser.parse_args()


def add_gpointnet_to_path(gpointnet_dir: str):
    gpointnet_dir = os.path.abspath(gpointnet_dir)
    if not os.path.isdir(gpointnet_dir):
        raise FileNotFoundError(f'GPointNet directory not found: {gpointnet_dir}')
    sys.path.insert(0, gpointnet_dir)
    sys.path.insert(0, os.path.join(gpointnet_dir, 'src'))


def load_gpointnet(checkpoint: str, gpointnet_dir: str, device: torch.device):
    add_gpointnet_to_path(gpointnet_dir)
    from src.model_point_torch import GPointNet

    model = GPointNet.load_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()
    model.energy_net.eval()
    return model


def patch_layernorm_num_points(model, target_num_point: int, device: torch.device):
    """Patch GPointNet LayerNorm layers so 1024-point HiT-ADV pairs can be evaluated.

    GPointNet often builds LayerNorm(config.num_point), e.g. 2048. HiT-ADV commonly
    saves 1024-point clouds. This patch mirrors the style used by GPointNet's own
    tools: replace learned per-point LayerNorm parameters by their mean repeated to
    the new point dimension.
    """
    patched = 0
    for layer in model.energy_net.local:
        if isinstance(layer, torch.nn.LayerNorm):
            old_shape = tuple(layer.normalized_shape)
            new_shape = (target_num_point,) if len(old_shape) == 1 else tuple(list(old_shape[:-1]) + [target_num_point])

            weight_mean = float(layer.weight.detach().mean().cpu().item())
            bias_mean = float(layer.bias.detach().mean().cpu().item())
            layer.normalized_shape = new_shape
            layer.weight = torch.nn.Parameter(torch.full(new_shape, weight_mean, device=device))
            layer.bias = torch.nn.Parameter(torch.full(new_shape, bias_mean, device=device))
            patched += 1
    return patched


def ensure_b_n_3(name: str, arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f'{name} must have shape (B,N,3), got {arr.shape}')
    if arr.shape[-1] == 3:
        return arr.astype(np.float32)
    if arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)
    raise ValueError(f'{name} must have xyz dimension of size 3, got {arr.shape}')


def adjust_num_points(points: np.ndarray, target_num_point: int) -> np.ndarray:
    """Deterministically adjust (B,N,3) to target point count.

    If N > target, take the first target points. If N < target, repeat points from
    the beginning. This is intentionally simple and reproducible; for the main
    planned workflow, target_num_point should equal the saved HiT-ADV point count.
    """
    bsz, num_point, channels = points.shape
    if channels != 3:
        raise ValueError(f'Expected 3 xyz channels, got {points.shape}')
    if num_point == target_num_point:
        return points
    if num_point > target_num_point:
        return points[:, :target_num_point, :]

    extra = target_num_point - num_point
    repeat_idx = np.arange(extra) % num_point
    return np.concatenate([points, points[:, repeat_idx, :]], axis=1)


def normalize_points(points: np.ndarray, mode: str, norm_min: float, norm_max: float) -> np.ndarray:
    if mode == 'none':
        return points
    if mode == 'provided_minmax':
        if norm_min is None or norm_max is None:
            raise ValueError('--norm_min and --norm_max are required for --normalize provided_minmax')
        denom = norm_max - norm_min
        if abs(denom) < 1e-12:
            raise ValueError('norm_max and norm_min are too close')
        return ((points - norm_min) / denom) * 2.0 - 1.0
    if mode == 'per_file_minmax':
        min_val = float(points.min())
        max_val = float(points.max())
        denom = max_val - min_val
        if abs(denom) < 1e-12:
            return points * 0.0
        return ((points - min_val) / denom) * 2.0 - 1.0
    raise ValueError(f'Unknown normalization mode: {mode}')


def compute_energy(model, points: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    energies: List[np.ndarray] = []
    for start in range(0, points.shape[0], batch_size):
        batch = points[start:start + batch_size]
        batch_t = torch.from_numpy(batch).float().to(device)
        batch_t = batch_t.transpose(1, 2).contiguous()  # (B,N,3) -> (B,3,N)
        with torch.no_grad():
            energy = model.energy_net(batch_t).squeeze(-1)
        energies.append(energy.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(energies, axis=0)


def append_npz_values(acc: Dict[str, List[np.ndarray]], data: Dict[str, np.ndarray]):
    for key, value in data.items():
        acc.setdefault(key, []).append(value)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model = load_gpointnet(args.checkpoint, args.gpointnet_dir, device)

    pair_files = sorted(glob.glob(os.path.join(args.pairs_dir, args.file_pattern)))
    if not pair_files:
        raise FileNotFoundError(f'No pair files found: {os.path.join(args.pairs_dir, args.file_pattern)}')

    first = np.load(pair_files[0])
    first_clean = ensure_b_n_3('clean', first['clean'])
    target_num_point = args.target_num_point if args.target_num_point > 0 else first_clean.shape[1]

    if not args.no_patch_layernorm:
        patched = patch_layernorm_num_points(model, target_num_point, device)
        print(f'Patched {patched} LayerNorm layer(s) to num_point={target_num_point}.')
    else:
        print('LayerNorm patch disabled; input point count must match checkpoint config.')

    csv_path = os.path.join(args.output_dir, 'energy_results.csv')
    npz_path = os.path.join(args.output_dir, 'energy_results.npz')

    aggregate: Dict[str, List[np.ndarray]] = {}
    rows: List[Dict[str, object]] = []

    for file_idx, pair_file in enumerate(pair_files):
        d = np.load(pair_file)

        # Apply target_label filter before energy computation
        if args.target_label is not None:
            keep_mask = (d['label'] == args.target_label)
            if not np.any(keep_mask):
                print(f'Skipping {os.path.basename(pair_file)}: no samples with label={args.target_label}')
                continue
        else:
            keep_mask = np.ones(d['label'].shape[0], dtype=bool)

        clean = adjust_num_points(ensure_b_n_3('clean', d['clean'][keep_mask]), target_num_point)
        adv = adjust_num_points(ensure_b_n_3('adv', d['adv'][keep_mask]), target_num_point)

        clean = normalize_points(clean, args.normalize, args.norm_min, args.norm_max).astype(np.float32)
        adv = normalize_points(adv, args.normalize, args.norm_min, args.norm_max).astype(np.float32)

        clean_energy = compute_energy(model, clean, device, args.batch_size)
        adv_energy = compute_energy(model, adv, device, args.batch_size)
        energy_diff = adv_energy - clean_energy

        labels = d['label'][keep_mask].astype(np.int64) if 'label' in d.files else np.full(clean.shape[0], -1, dtype=np.int64)
        ori_pred = d['ori_pred'][keep_mask].astype(np.int64) if 'ori_pred' in d.files else np.full(clean.shape[0], -1, dtype=np.int64)
        adv_pred = d['adv_pred'][keep_mask].astype(np.int64) if 'adv_pred' in d.files else np.full(clean.shape[0], -1, dtype=np.int64)
        attack_success = d['attack_success'][keep_mask].astype(bool) if 'attack_success' in d.files else np.zeros(clean.shape[0], dtype=bool)

        sidecar_name = os.path.basename(pair_file).replace('.npz', '_energy.npz')
        np.savez_compressed(
            os.path.join(args.output_dir, sidecar_name),
            clean_energy=clean_energy,
            adv_energy=adv_energy,
            energy_diff=energy_diff,
            label=labels,
            ori_pred=ori_pred,
            adv_pred=adv_pred,
            attack_success=attack_success,
            source_file=np.array([os.path.basename(pair_file)] * clean.shape[0]),
        )

        append_npz_values(aggregate, {
            'clean_energy': clean_energy,
            'adv_energy': adv_energy,
            'energy_diff': energy_diff,
            'label': labels,
            'ori_pred': ori_pred,
            'adv_pred': adv_pred,
            'attack_success': attack_success,
            'source_file': np.array([os.path.basename(pair_file)] * clean.shape[0]),
        })

        for sample_idx in range(clean.shape[0]):
            rows.append({
                'source_file': os.path.basename(pair_file),
                'file_index': file_idx,
                'sample_index': sample_idx,
                'label': int(labels[sample_idx]),
                'ori_pred': int(ori_pred[sample_idx]),
                'adv_pred': int(adv_pred[sample_idx]),
                'attack_success': bool(attack_success[sample_idx]),
                'clean_energy': float(clean_energy[sample_idx]),
                'adv_energy': float(adv_energy[sample_idx]),
                'energy_diff': float(energy_diff[sample_idx]),
            })
        print(f'Processed {file_idx + 1}/{len(pair_files)}: {os.path.basename(pair_file)}')

    fieldnames = [
        'source_file',
        'file_index',
        'sample_index',
        'label',
        'ori_pred',
        'adv_pred',
        'attack_success',
        'clean_energy',
        'adv_energy',
        'energy_diff',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if aggregate:
        np.savez_compressed(npz_path, **{key: np.concatenate(value, axis=0) for key, value in aggregate.items()})
    else:
        np.savez_compressed(
            npz_path,
            clean_energy=np.array([], dtype=np.float32),
            adv_energy=np.array([], dtype=np.float32),
            energy_diff=np.array([], dtype=np.float32),
            label=np.array([], dtype=np.int64),
            ori_pred=np.array([], dtype=np.int64),
            adv_pred=np.array([], dtype=np.int64),
            attack_success=np.array([], dtype=bool),
            source_file=np.array([], dtype=str),
        )
        print('No samples matched the requested filters; wrote empty result files.')
    print(f'Saved CSV: {csv_path}')
    print(f'Saved NPZ: {npz_path}')


if __name__ == '__main__':
    main()
