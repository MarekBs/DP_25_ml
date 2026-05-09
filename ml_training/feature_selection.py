import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

CORR_THRESHOLD = 0.95

OPTIMAL_K = {
    "pickup": {"SVM": 46, "Random Forest": 32, "XGBoost": 60, "KNN": 48},
    "table":  {"SVM": 92, "Random Forest": 22, "XGBoost": 30, "KNN": 38},
    "swipe":  {"SVM": 22, "Random Forest": 18, "XGBoost": 14, "KNN": 18},
    "zoom":   {"SVM": 10, "Random Forest": 12, "XGBoost": 20, "KNN": 10},
}


def correlation_filter(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    df_new = df.drop(columns=to_drop)
    kept = df_new.columns.tolist()
    print(f"[1/2] CorrFilter    : {X.shape[1]:>4} -> {len(kept):>4} features  (odstranených {len(to_drop)})")
    return df_new.values, kept


def compute_importances(X, y, feature_names):
    users = np.unique(y)
    importances = np.zeros(X.shape[1])
    X_sc = StandardScaler().fit_transform(X)
    for user in users:
        y_bin = (y == user).astype(int)
        if y_bin.sum() < 2:
            continue
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_sc, y_bin)
        importances += rf.feature_importances_
    importances /= len(users)
    return importances


def rf_importance_filter(X, y, feature_names, k):
    print(f"[2/2] RF Importance : pocitam cez {len(np.unique(y))} pouzivatelov ...")
    importances = compute_importances(X, y, feature_names)

    top_idx = np.argsort(importances)[::-1][:k]
    kept = [feature_names[i] for i in top_idx]
    X_new = X[:, top_idx]

    print(f"           Vysledok: {X.shape[1]:>4} -> {X_new.shape[1]:>4} features  (top {k})")
    print(f"\n{'#':<5} {'Feature':<40} {'Importance':>10}")
    print("-" * 57)
    for rank, (name, imp) in enumerate(zip(kept, importances[top_idx]), 1):
        print(f"{rank:<5} {name:<40} {imp:>10.4f}")
    print()

    return X_new, kept


def select_features(X, y, feature_names, gesture=None, model="Random Forest", mode="none"):
    if mode == "none":
        print("[Feature selection: vypnuta]\n")
        return X, feature_names

    print("\n" + "=" * 57)
    n_orig = X.shape[1]

    if mode == "corr":
        print("FEATURE SELECTION  (korelacny filter)")
        print("=" * 57)
        X, feature_names = correlation_filter(X, feature_names)

    elif mode == "full":
        k = OPTIMAL_K.get(gesture, {}).get(model)
        if k is None:
            print(f"FEATURE SELECTION  (korelacny filter, k nezname pre {gesture}/{model})")
            print("=" * 57)
            X, feature_names = correlation_filter(X, feature_names)
        else:
            print(f"FEATURE SELECTION  (gesture={gesture}, model={model}, k={k})")
            print("=" * 57)
            X, feature_names = correlation_filter(X, feature_names)
            X, feature_names = rf_importance_filter(X, y, feature_names, k)

    print(f"Celkovo: {n_orig} -> {len(feature_names)} features\n")
    return X, feature_names
