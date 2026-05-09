#!/usr/bin/env python3
import sys, importlib
import numpy as np
from feature_selection import correlation_filter, compute_importances
from training import train_and_evaluate

K_VALUES = list(range(2, 39, 2)) + [None]

GESTURE_CONFIG = {
    "swipe":  {"module": "train_swipe",  "data_dir": "./data_swipe",      "train_kwargs": {}},
    "walk":   {"module": "train_walk",   "data_dir": "./data_walk",       "train_kwargs": {"min_samples": 5}},
    "table":  {"module": "train_table",  "data_dir": "./data_stol",       "train_kwargs": {}},
    "pickup": {"module": "train_pickup", "data_dir": "./data_zdvihnutie", "train_kwargs": {}},
    "zoom":   {"module": "train_zoom",   "data_dir": "./data_zoom",       "train_kwargs": {}},
}

if len(sys.argv) < 2 or sys.argv[1] not in GESTURE_CONFIG:
    print(f"Použitie: python find_optimal_k.py <gesto>")
    print(f"Gestá: {', '.join(GESTURE_CONFIG)}")
    sys.exit(1)

GESTURE = sys.argv[1]
cfg = GESTURE_CONFIG[GESTURE]
mod = importlib.import_module(cfg["module"])

print(f"\nNačítavam dataset: {GESTURE}")
X, y, feature_names = mod.load_dataset(cfg["data_dir"])

print("\nCorrelation filter...")
X_corr, names_corr = correlation_filter(X, feature_names)

print("Počítam RF importancie (raz pre všetky k)...")
importances = compute_importances(X_corr, y, names_corr)
ranked_idx = np.argsort(importances)[::-1]
n_avail = len(names_corr)

summary = []
for k in K_VALUES:
    if k is None or k >= n_avail:
        X_k, names_k, k_label = X_corr, names_corr, f"all({n_avail})"
    else:
        idx = ranked_idx[:k]
        X_k = X_corr[:, idx]
        names_k = [names_corr[i] for i in idx]
        k_label = str(k)

    print(f"\n{'='*50}  k={k_label}")
    results = train_and_evaluate(X_k, y, names_k, output_pkl=None, **cfg["train_kwargs"])

    best = max(results, key=lambda m: np.mean(results[m]["accs"]) if results[m]["accs"] else 0)
    r = results[best]
    summary.append((k_label, best, np.mean(r["accs"]), np.mean(r["eers"]),
                    np.mean(r["aucs"]), np.mean(r["f1s"])))

print(f"\n\n{'='*58}")
print(f"SÚHRN  —  gesto: {GESTURE}")
print(f"{'='*58}")
print(f"{'k':<12} {'Model':<16} {'Acc':>6} {'EER':>6} {'AUC':>6} {'F1':>6}")
print("-" * 58)
for k_label, model, acc, eer, auc, f1 in summary:
    print(f"{k_label:<12} {model:<16} {acc:>6.3f} {eer:>6.3f} {auc:>6.3f} {f1:>6.3f}")
