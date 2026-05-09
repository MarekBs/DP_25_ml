#!/usr/bin/env python3
import importlib, json
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from feature_selection import correlation_filter, compute_importances
from training import train_and_evaluate

N_TRIALS = {
    "SVM":           100,
    "Random Forest": 50,
    "XGBoost":       75,
    "KNN":           100,
}

OPTIMAL_K = {
    "pickup": {"SVM": 46, "Random Forest": 32, "XGBoost": 60, "KNN": 48},
    "table":  {"SVM": 92, "Random Forest": 22, "XGBoost": 30, "KNN": 38},
    "swipe":  {"SVM": 22, "Random Forest": 18, "XGBoost": 14, "KNN": 18},
    "zoom":   {"SVM": 10, "Random Forest": 12, "XGBoost": 20, "KNN": 10},
}

GESTURE_CONFIG = {
    "swipe":  {"module": "train_swipe",  "data_dir": "./data_swipe",      "train_kwargs": {}},
    "walk":   {"module": "train_walk",   "data_dir": "./data_walk",       "train_kwargs": {"min_samples": 5}},
    "table":  {"module": "train_table",  "data_dir": "./data_stol",       "train_kwargs": {}},
    "pickup": {"module": "train_pickup", "data_dir": "./data_zdvihnutie", "train_kwargs": {}},
    "zoom":   {"module": "train_zoom",   "data_dir": "./data_zoom",       "train_kwargs": {}},
}

MODELS = [ "KNN"]


def suggest_params(trial, model_name):
    if model_name == "SVM":
        kernel     = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        gamma_type = trial.suggest_categorical("gamma_type", ["scale", "auto", "float"])
        gamma      = trial.suggest_float("gamma_val", 1e-6, 10.0, log=True) if gamma_type == "float" else gamma_type
        params = {
            "kernel": kernel,
            "C":      trial.suggest_float("C", 1e-3, 1e4, log=True),
            "gamma":  gamma,
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        return params

    if model_name == "Random Forest":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth":         trial.suggest_categorical("max_depth", [None, 20, 30]),
            "min_samples_leaf":  trial.suggest_categorical("min_samples_leaf", [1, 2]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5]),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        }

    if model_name == "XGBoost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        }

    if model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "metric":      trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm":   trial.suggest_categorical("algorithm", ["auto", "ball_tree", "brute"]),
        }


print("Predpočítavam datasety a importancie...")
CACHE = {}
for gesture, model_k in OPTIMAL_K.items():
    cfg = GESTURE_CONFIG[gesture]
    mod = importlib.import_module(cfg["module"])
    print(f"  {gesture}...")
    X, y, feature_names = mod.load_dataset(cfg["data_dir"])
    X_corr, names_corr = correlation_filter(X, feature_names)
    importances = compute_importances(X_corr, y, names_corr)
    ranked_idx = np.argsort(importances)[::-1]
    n_avail = len(names_corr)
    for model_name in MODELS:
        k = min(model_k[model_name], n_avail)
        idx = ranked_idx[:k]
        CACHE[(gesture, model_name)] = (
            X_corr[:, idx],
            y,
            [names_corr[i] for i in idx],
            cfg["train_kwargs"],
        )

best_results = {}

for model_name in MODELS:
    print(f"\n{'='*50}")
    n_trials = N_TRIALS.get(model_name, 50)
    print(f"Optuna: {model_name}  ({n_trials} trials)")
    print(f"{'='*50}")

    def objective(trial):
        params = suggest_params(trial, model_name)
        eers = []
        for gesture in OPTIMAL_K:
            X_k, y, names_k, train_kwargs = CACHE[(gesture, model_name)]
            results = train_and_evaluate(
                X_k, y, names_k,
                only_models=[model_name],
                params={model_name: params},
                verbose=False,
                **train_kwargs,
            )
            eers.extend(results[model_name]["eers"])
        return np.mean(eers) if eers else 1.0

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[lambda study, trial: print(
            f"  Trial {trial.number+1:>3}/{n_trials}  "
            f"EER={trial.value:.4f}  best={study.best_value:.4f}"
        )],
    )

    best_results[model_name] = {
        "params": study.best_params,
        "eer":    study.best_value,
    }
    print(f"\n→ Najlepšie: EER={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

print(f"\n\n{'='*60}")
print("SÚHRN NAJLEPŠÍCH PARAMETROV")
print(f"{'='*60}")
for model_name, info in best_results.items():
    print(f"\n{model_name}  (EER={info['eer']:.4f})")
    for k, v in info["params"].items():
        print(f"  {k}: {v}")

with open("best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_results, f, indent=2, default=str)
print("\nUložené do best_params.json")
