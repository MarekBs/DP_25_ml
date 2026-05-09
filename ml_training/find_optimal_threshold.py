#!/usr/bin/env python3
import importlib, json
import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from feature_selection import correlation_filter, compute_importances
from training import make_models

N_FOLDS     = 5
MIN_SAMPLES = 2

OPTIMAL_K = {
    "pickup": {"SVM": 46, "Random Forest": 32, "XGBoost": 60, "KNN": 48},
    "table":  {"SVM": 92, "Random Forest": 22, "XGBoost": 30, "KNN": 38},
    "swipe":  {"SVM": 22, "Random Forest": 18, "XGBoost": 14, "KNN": 18},
    "zoom":   {"SVM": 10, "Random Forest": 12, "XGBoost": 20, "KNN": 10},
}

GESTURE_CONFIG = {
    "swipe":  {"module": "train_swipe",  "data_dir": "../data/swipe",      "min_samples": 2},
    "table":  {"module": "train_table",  "data_dir": "../data/stol",       "min_samples": 2},
    "pickup": {"module": "train_pickup", "data_dir": "../data/zdvihnutie", "min_samples": 2},
    "zoom":   {"module": "train_zoom",   "data_dir": "../data/zoom",       "min_samples": 2},
}

MODELS = list(make_models().keys())


def oof_threshold(model, X_trainval, y_trainval, cv):
    oof_proba = np.zeros(len(y_trainval))
    for train_idx, val_idx in cv.split(X_trainval, y_trainval):
        m = clone(model)
        m.fit(X_trainval[train_idx], y_trainval[train_idx])
        oof_proba[val_idx] = m.predict_proba(X_trainval[val_idx])[:, 1]

    if len(np.unique(y_trainval)) < 2:
        return 0.5

    fpr_c, tpr_c, thresh_c = roc_curve(y_trainval, oof_proba)
    fnr_c = 1 - tpr_c
    idx   = np.argmin(np.abs(fpr_c - fnr_c))
    return float(thresh_c[idx])


def evaluate_with_threshold(X, y, model_name, min_samples=2):
    users  = np.unique(y)
    stats  = {m: [] for m in ["threshold", "acc", "far", "frr", "eer",
                               "prec", "rec", "f1", "auc", "cv_auc", "hits", "miss"]}

    for target_user in users:
        y_bin = (y == target_user).astype(int)

        rng     = np.random.default_rng(42)
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx)]

        if len(pos_idx) < min_samples:
            continue

        n_pos_test = max(1, int(round(len(pos_idx) * 0.30)))
        n_neg_test = max(1, int(round(len(neg_idx) * 0.30)))

        test_idx     = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
        trainval_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

        X_trainval, y_trainval = X[trainval_idx], y_bin[trainval_idx]
        X_test,     y_test     = X[test_idx],     y_bin[test_idx]

        if len(np.unique(y_trainval)) < 2:
            continue

        n_splits = min(N_FOLDS, int(np.min(np.bincount(y_trainval))))
        if n_splits < 2:
            continue

        cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = make_models()[model_name]

        if model_name == "KNN":
            clf = model.named_steps["clf"]
            min_cv_size = len(X_trainval) * (n_splits - 1) // n_splits
            clf.n_neighbors = min(clf.n_neighbors, min_cv_size - 1)

        opt_thresh = oof_threshold(model, X_trainval, y_trainval, cv)

        cv_auc = cross_val_score(model, X_trainval, y_trainval,
                                 cv=cv, scoring="roc_auc").mean()

        model.fit(X_trainval, y_trainval)
        y_proba = model.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) < 2:
            continue

        fpr_c, tpr_c, _ = roc_curve(y_test, y_proba)
        fnr_c = 1 - tpr_c
        eer   = float((fpr_c + fnr_c)[np.argmin(np.abs(fpr_c - fnr_c))]) / 2
        auc   = roc_auc_score(y_test, y_proba)

        y_pred = (y_proba >= opt_thresh).astype(int)
        TP = int(((y_pred == 1) & (y_test == 1)).sum())
        FP = int(((y_pred == 1) & (y_test == 0)).sum())
        TN = int(((y_pred == 0) & (y_test == 0)).sum())
        FN = int(((y_pred == 0) & (y_test == 1)).sum())

        stats["threshold"].append(opt_thresh)
        stats["acc"].append(accuracy_score(y_test, y_pred))
        stats["far"].append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
        stats["frr"].append(FN / (FN + TP) if (FN + TP) > 0 else 0.0)
        stats["eer"].append(eer)
        stats["prec"].append(precision_score(y_test, y_pred, zero_division=0))
        stats["rec"].append(recall_score(y_test, y_pred, zero_division=0))
        stats["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        stats["auc"].append(auc)
        stats["cv_auc"].append(cv_auc)
        stats["hits"].append(TP + TN)
        stats["miss"].append(FP + FN)

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
    result["hits"] = int(sum(stats["hits"]))
    result["miss"] = int(sum(stats["miss"]))
    return result


print("Načítavam datasety a počítam optimálne prahy...")
all_results = {}

for gesture, cfg in GESTURE_CONFIG.items():
    print(f"\n{'='*75}")
    print(f"Gesto: {gesture}")
    print(f"{'='*75}")

    mod = importlib.import_module(cfg["module"])
    X, y, feature_names = mod.load_dataset(cfg["data_dir"])
    X_corr, names_corr  = correlation_filter(X, feature_names)
    importances         = compute_importances(X_corr, y, names_corr)
    ranked_idx          = np.argsort(importances)[::-1]
    n_avail             = len(names_corr)
    all_results[gesture] = {}

    hdr = f"{'k':<5} {'Model':<16} {'Prah':>5} {'Acc':>6} {'FAR':>6} {'FRR':>6} {'EER':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-AUC':>8} {'Hits':>6} {'Miss':>6}"
    print(hdr)
    print("-" * len(hdr))

    for model_name in MODELS:
        k   = min(OPTIMAL_K[gesture][model_name], n_avail)
        idx = ranked_idx[:k]
        X_k = X_corr[:, idx]

        r = evaluate_with_threshold(X_k, y, model_name, cfg["min_samples"])
        all_results[gesture][model_name] = {"k": k, **r}

        print(f"{k:<5} {model_name:<16} {r['threshold']:>5.3f} {r['acc']:>6.3f} "
              f"{r['far']:>6.3f} {r['frr']:>6.3f} {r['eer']:>6.3f} "
              f"{r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f} "
              f"{r['auc']:>6.3f} {r['cv_auc']:>8.3f} "
              f"{int(r['hits']):>6} {int(r['miss']):>6}")

with open("optimal_thresholds.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print("\nUložené do optimal_thresholds.json")
