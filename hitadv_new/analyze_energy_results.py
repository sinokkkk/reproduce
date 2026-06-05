"""Analyze energy_results.csv and energy_results.npz with summary statistics.

Prints overall, successful-only, and per-label summaries for:
- clean_energy (mean ± std)
- adv_energy (mean ± std)
- energy_diff (mean ± std)
- P(delta_E > 0)  -- fraction where adv_energy > clean_energy
- P(delta_E < 0)  -- fraction where adv_energy < clean_energy

Handles zero rows gracefully (prints "no data" instead of crashing).
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze GPointNet energy results from CSV and NPZ files."
    )
    parser.add_argument('--csv_path', type=str, default='energy_results/energy_results.csv',
                        help='path to energy_results.csv')
    parser.add_argument('--npz_path', type=str, default='energy_results/energy_results.npz',
                        help='path to energy_results.npz (fallback if CSV unavailable)')
    parser.add_argument('--target_label', type=int, default=None,
                        help='optional label filter before reporting statistics, e.g. 8 for chair')
    return parser.parse_args()


def load_rows(csv_path: str, npz_path: str) -> List[Dict[str, Any]]:
    """Load rows from CSV, falling back to NPZ if CSV is missing."""
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            return rows

    # Fall back to NPZ
    if not os.path.exists(npz_path):
        print(f"Neither CSV ({csv_path}) nor NPZ ({npz_path}) found.")
        return []

    print(f"CSV not found or empty; loading from NPZ: {npz_path}")
    data = np.load(npz_path)
    n = len(data['clean_energy'])
    rows = []
    for i in range(n):
        row: Dict[str, Any] = {}
        for key in data.files:
            val = data[key][i]
            if isinstance(val, np.bool_):
                row[key] = bool(val)
            elif isinstance(val, np.integer):
                row[key] = int(val)
            elif isinstance(val, np.floating):
                row[key] = float(val)
            elif isinstance(val, (bytes, np.bytes_)):
                row[key] = val.decode('utf-8') if isinstance(val, (bytes, np.bytes_)) else str(val)
            else:
                row[key] = str(val) if isinstance(val, np.ndarray) and val.ndim == 0 else val
        rows.append(row)
    return rows


def print_summary(title: str, rows: List[Dict[str, Any]]):
    """Print mean ± std for energy columns and P(delta_E > 0 / < 0)."""
    n = len(rows)
    print(f"\n{'=' * 60}")
    print(f"  {title}  (n={n})")
    print(f"{'=' * 60}")

    if n == 0:
        print("  No data.")
        return

    clean_vals = np.array([float(r.get('clean_energy', np.nan)) for r in rows])
    adv_vals = np.array([float(r.get('adv_energy', np.nan)) for r in rows])
    diff_vals = np.array([float(r.get('energy_diff', np.nan)) for r in rows])

    clean_valid = clean_vals[~np.isnan(clean_vals)]
    adv_valid = adv_vals[~np.isnan(adv_vals)]
    diff_valid = diff_vals[~np.isnan(diff_vals)]

    def _fmt(arr):
        if len(arr) == 0:
            return "N/A"
        return f"{float(np.mean(arr)):.6f} ± {float(np.std(arr)):.6f}"

    print(f"  clean_energy:   {_fmt(clean_valid)}")
    print(f"  adv_energy:     {_fmt(adv_valid)}")
    print(f"  energy_diff:    {_fmt(diff_valid)}")

    if len(diff_valid) > 0:
        p_pos = float(np.mean(diff_valid > 0))
        p_neg = float(np.mean(diff_valid < 0))
        print(f"  P(delta_E > 0): {p_pos:.4f}  ({int(p_pos * len(diff_valid))}/{len(diff_valid)})")
        print(f"  P(delta_E < 0): {p_neg:.4f}  ({int(p_neg * len(diff_valid))}/{len(diff_valid)})")
    else:
        print("  P(delta_E > 0): N/A")
        print("  P(delta_E < 0): N/A")


def main():
    args = parse_args()
    rows = load_rows(args.csv_path, args.npz_path)

    if not rows:
        print("No rows to analyze.")
        return

    if args.target_label is not None:
        rows = [r for r in rows if r.get('label') is not None and int(r['label']) == args.target_label]
        if not rows:
            print(f"No rows matched target_label={args.target_label}.")
            return
        print(f"Filtered to target_label={args.target_label}; rows={len(rows)}")

    # ── Overall summary ──
    print_summary("OVERALL", rows)

    # ── Successful attacks only ──
    success_rows = []
    for r in rows:
        val = r.get('attack_success', '')
        if isinstance(val, str):
            is_success = val.lower() in ('true', '1')
        elif isinstance(val, bool):
            is_success = val
        elif isinstance(val, (int, np.integer)):
            is_success = bool(val)
        else:
            is_success = False
        if is_success:
            success_rows.append(r)

    if success_rows:
        print_summary("SUCCESSFUL ATTACKS ONLY", success_rows)
    else:
        print("\n--- SUCCESSFUL ATTACKS ONLY ---")
        print("  No successful attacks found.")

    # ── Per-label summary ──
    label_set = set()
    for r in rows:
        lbl = r.get('label')
        if lbl is not None:
            label_set.add(int(lbl))

    if label_set:
        print(f"\n{'=' * 60}")
        print("  PER-LABEL SUMMARY")
        print(f"{'=' * 60}")
        for lbl in sorted(label_set):
            lbl_rows = [r for r in rows if r.get('label') is not None and int(r['label']) == lbl]
            print_summary(f"Label {lbl}", lbl_rows)
    else:
        print("\nNo label information available for per-label breakdown.")


if __name__ == '__main__':
    main()
