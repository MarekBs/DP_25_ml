import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import joblib

USE_FEATURE_SELECTION = False
TOP_N_FEATURES        = 40
N_FOLDS               = 5

SENSOR_MARGIN_NS = 100_000_000
SENSOR_COLS      = ["input", "session_id", "timestamp_ns", "x", "y", "z"]


def sanitize_path(name):
    return re.sub(r'[<>:"/\\|?*()]', '_', name)


def parse_sensor_file(filepath):
    df = pd.read_csv(filepath, header=None, names=SENSOR_COLS)
    for col in ["timestamp_ns", "x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["timestamp_ns", "x", "y", "z"], inplace=True)
    df.sort_values("timestamp_ns", inplace=True)
    return df.reset_index(drop=True)


def clip_sensor_to_segment(sensor_df, start_ts, end_ts, margin_ns=SENSOR_MARGIN_NS):
    if sensor_df is None or sensor_df.empty:
        return None
    clip = sensor_df[(sensor_df["timestamp_ns"] >= start_ts - margin_ns) &
                     (sensor_df["timestamp_ns"] <= end_ts   + margin_ns)]
    return clip.reset_index(drop=True) if len(clip) >= 3 else None


def sensor_paths_for(touch_path):
    base   = touch_path.name[:-len("_touch.csv")]
    parent = touch_path.parent
    return parent / f"{base}_sensor_accelerometer.csv", \
           parent / f"{base}_sensor_gyroscope.csv"


def sensor_axis_features(v, prefix):
    feats = {}
    feats[f"{prefix}_min"]      = float(np.min(v))
    feats[f"{prefix}_max"]      = float(np.max(v))
    feats[f"{prefix}_mean"]     = float(np.mean(v))
    feats[f"{prefix}_var"]      = float(np.var(v))
    feats[f"{prefix}_std"]      = float(np.std(v))
    feats[f"{prefix}_median"]   = float(np.median(v))
    feats[f"{prefix}_skewness"] = float(skew(v))        if len(v) > 2 else 0.0
    feats[f"{prefix}_kurtosis"] = float(sp_kurtosis(v)) if len(v) > 3 else 0.0
    feats[f"{prefix}_q1"]       = float(np.percentile(v, 25))
    feats[f"{prefix}_q3"]       = float(np.percentile(v, 75))
    feats[f"{prefix}_iqr"]      = feats[f"{prefix}_q3"] - feats[f"{prefix}_q1"]
    feats[f"{prefix}_rms"]      = float(np.sqrt(np.mean(v**2)))
    feats[f"{prefix}_zero_crossing"] = int(np.sum(np.diff(np.sign(v - np.mean(v))) != 0))
    feats[f"{prefix}_energy"]   = float(np.sum(v**2))

    peaks, _ = find_peaks(v)
    pv = v[peaks] if len(peaks) > 0 else np.array([0.0])
    feats[f"{prefix}_n_peaks"]   = int(len(peaks))
    feats[f"{prefix}_peak_min"]  = float(np.min(pv))
    feats[f"{prefix}_peak_max"]  = float(np.max(pv))
    feats[f"{prefix}_peak_mean"] = float(np.mean(pv))

    fft = np.abs(np.fft.rfft(v))
    feats[f"{prefix}_fft_sum"] = float(np.sum(fft))
    return feats


def extract_sensor_features(sensor_df, prefix):
    if sensor_df is None or len(sensor_df) < 3:
        feats = {}
        for axis in ["x", "y", "z", "mag"]:
            for k in ["min", "max", "mean", "var", "std", "median", "skewness", "kurtosis",
                      "q1", "q3", "iqr", "rms", "zero_crossing", "energy",
                      "n_peaks", "peak_min", "peak_max", "peak_mean", "fft_sum"]:
                feats[f"{prefix}_{axis}_{k}"] = 0.0
        for k in ["cor_xy", "cor_xz", "cor_yz", "duration_ms", "n_samples"]:
            feats[f"{prefix}_{k}"] = 0.0
        return feats

    x = sensor_df["x"].values.astype(np.float64)
    y = sensor_df["y"].values.astype(np.float64)
    z = sensor_df["z"].values.astype(np.float64)
    mag = np.sqrt(x**2 + y**2 + z**2)

    feats = {}
    feats.update(sensor_axis_features(x,   f"{prefix}_x"))
    feats.update(sensor_axis_features(y,   f"{prefix}_y"))
    feats.update(sensor_axis_features(z,   f"{prefix}_z"))
    feats.update(sensor_axis_features(mag, f"{prefix}_mag"))

    feats[f"{prefix}_cor_xy"] = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    feats[f"{prefix}_cor_xz"] = float(np.corrcoef(x, z)[0, 1]) if np.std(x) > 0 and np.std(z) > 0 else 0.0
    feats[f"{prefix}_cor_yz"] = float(np.corrcoef(y, z)[0, 1]) if np.std(y) > 0 and np.std(z) > 0 else 0.0

    t_ms = (sensor_df["timestamp_ns"].values - sensor_df["timestamp_ns"].values[0]) / 1e6
    feats[f"{prefix}_duration_ms"] = float(t_ms[-1] - t_ms[0]) if len(t_ms) > 1 else 0.0
    feats[f"{prefix}_n_samples"]   = int(len(sensor_df))
    return feats


def cross_sensor_features(acc_df, gyr_df):
    feats = {}
    if acc_df is None or gyr_df is None or len(acc_df) < 3 or len(gyr_df) < 3:
        for axis in ["x", "y", "z"]:
            feats[f"acc_gyr_cor_{axis}"] = 0.0
        return feats

    a = acc_df[["x", "y", "z"]].values
    g = gyr_df[["x", "y", "z"]].values
    n = min(len(a), len(g))
    for i, axis in enumerate(["x", "y", "z"]):
        a_i, g_i = a[:n, i], g[:n, i]
        if n >= 3 and np.std(a_i) > 0 and np.std(g_i) > 0:
            feats[f"acc_gyr_cor_{axis}"] = float(np.corrcoef(a_i, g_i)[0, 1])
        else:
            feats[f"acc_gyr_cor_{axis}"] = 0.0
    return feats


def make_models():
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                   eval_metric="logloss", random_state=42, n_jobs=-1))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
    }



def select_features(X, y, feature_names):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:TOP_N_FEATURES]
    print(f"Feature selection: top {TOP_N_FEATURES} z {X.shape[1]} príznakov")
    return X[:, top_idx], [feature_names[i] for i in top_idx]


def train_and_evaluate(X, y, feature_names, output_pkl):
    if USE_FEATURE_SELECTION:
        X, feature_names = select_features(X, y, feature_names)

    users       = np.unique(y)
    model_names = list(make_models().keys())
    results     = {name: {"fars": [], "frrs": [], "eers": [], "aucs": [], "accs": [],
                          "precs": [], "recs": [], "f1s": [], "hits": [], "misses": [],
                          "cv_aucs": []}
                   for name in model_names}
    best_models = {name: {} for name in model_names}

    for target_user in users:
        y_bin = (y == target_user).astype(int)

        rng     = np.random.default_rng(42)
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx)]

        if len(pos_idx) < 2:
            print(f"  [SKIP] {target_user}: príliš málo pozitívnych vzoriek")
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
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for name, model in make_models().items():
            cv_auc = cross_val_score(model, X_trainval, y_trainval,
                                     cv=cv, scoring="roc_auc").mean()
            model.fit(X_trainval, y_trainval)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            TP = int(((y_pred == 1) & (y_test == 1)).sum())
            FP = int(((y_pred == 1) & (y_test == 0)).sum())
            TN = int(((y_pred == 0) & (y_test == 0)).sum())
            FN = int(((y_pred == 0) & (y_test == 1)).sum())

            FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

            if len(np.unique(y_test)) > 1:
                fpr_c, tpr_c, _ = roc_curve(y_test, y_proba)
                fnr_c = 1 - tpr_c
                eer = float((fpr_c + fnr_c)[np.argmin(np.abs(fpr_c - fnr_c))]) / 2
                auc = roc_auc_score(y_test, y_proba)
            else:
                eer = auc = 0.0

            results[name]["cv_aucs"].append(cv_auc)
            results[name]["fars"].append(FAR)
            results[name]["frrs"].append(FRR)
            results[name]["eers"].append(eer)
            results[name]["aucs"].append(auc)
            results[name]["accs"].append(accuracy_score(y_test, y_pred))
            results[name]["precs"].append(precision_score(y_test, y_pred, zero_division=0))
            results[name]["recs"].append(recall_score(y_test, y_pred, zero_division=0))
            results[name]["f1s"].append(f1_score(y_test, y_pred, zero_division=0))
            results[name]["hits"].append(int((y_pred == y_test).sum()))
            results[name]["misses"].append(int((y_pred != y_test).sum()))
            best_models[name][target_user] = model

    hdr = f"\n{'Model':<20} {'Acc':>6} {'FAR':>6} {'FRR':>6} {'EER':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-AUC':>8} {'Hits':>8} {'Miss':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in model_names:
        r = results[name]
        if not r["accs"]:
            continue
        print(f"{name:<20} {np.mean(r['accs']):>6.3f} {np.mean(r['fars']):>6.3f} "
              f"{np.mean(r['frrs']):>6.3f} {np.mean(r['eers']):>6.3f} {np.mean(r['precs']):>6.3f} "
              f"{np.mean(r['recs']):>6.3f} {np.mean(r['f1s']):>6.3f} {np.mean(r['aucs']):>6.3f} "
              f"{np.mean(r['cv_aucs']):>8.3f} {sum(r['hits']):>8} {sum(r['misses']):>8}")

    best_name = max(model_names, key=lambda k: np.mean(results[k]["accs"]) if results[k]["accs"] else 0)
    joblib.dump({
        "models": best_models[best_name],
        "feature_names": feature_names,
        "model_type": best_name,
    }, output_pkl)
    print(f"\nNajlepší model: {best_name} "
          f"(avg acc={np.mean(results[best_name]['accs']):.4f}) -> {output_pkl}")
