import re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks
from training import train_and_evaluate
from feature_selection import select_features

FIREBASE_BUCKET       = "dpapp-18ab8.firebasestorage.app"
GESTURE_PATH          = "sensors_logs_behametrics/Položenie na stôl"
LOCAL_DATA_DIR        = str(Path(__file__).parent.parent / "data" / "stol")
SERVICE_ACCOUNT       = "serviceAccountKey.json"


RE_ACCEL = re.compile(r"log(\d+)_sensor_accelerometer\.csv", re.IGNORECASE)
RE_GYRO  = re.compile(r"log(\d+)_sensor_gyroscope\.csv",     re.IGNORECASE)


def sanitize_path(name):
    return re.sub(r'[<>:"/\\|?*()]', '_', name)


def download_from_firebase(local_dir):
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage

    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            credentials.Certificate(SERVICE_ACCOUNT),
            {"storageBucket": FIREBASE_BUCKET}
        )

    bucket = fb_storage.bucket()
    blobs  = list(bucket.list_blobs(prefix=GESTURE_PATH + "/"))
    print(f"Firebase: najdených {len(blobs)} suborov")

    for blob in blobs:
        parts = blob.name[len(GESTURE_PATH) + 1:].split("/", 1)
        if len(parts) != 2 or not parts[1].endswith(".csv"):
            continue
        user_id, filename = parts
        safe_user_id = sanitize_path(user_id)
        dest = Path(local_dir) / safe_user_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"  Stahujem: {filename}")
            blob.download_to_filename(str(dest))

    print("Hotovo.")


def parse_file(filepath):
    df = pd.read_csv(filepath, header=None,
                     names=["sensor", "user_id", "timestamp_ns", "x", "y", "z"])
    df = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def trim_gesture(df, smooth=5, padding=10):
    mag       = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    mag       = mag.rolling(window=smooth, center=True, min_periods=1).mean()
    threshold = 0.15 * np.percentile(mag, 95)
    active    = np.where(mag > threshold)[0]
    if len(active) == 0:
        return df
    start = max(0, active[0] - padding)
    end   = min(len(df) - 1, active[-1] + padding)
    return df.iloc[start:end + 1].reset_index(drop=True)


def axis_features(v, prefix):
    feats = {}
    feats[f"{prefix}_min"]      = np.min(v)
    feats[f"{prefix}_max"]      = np.max(v)
    feats[f"{prefix}_mean"]     = np.mean(v)
    feats[f"{prefix}_var"]      = np.var(v)
    feats[f"{prefix}_std"]      = np.std(v)
    feats[f"{prefix}_median"]   = np.median(v)
    feats[f"{prefix}_skewness"] = skew(v)
    feats[f"{prefix}_kurtosis"] = sp_kurtosis(v)
    feats[f"{prefix}_q1"]       = np.percentile(v, 25)
    feats[f"{prefix}_q3"]       = np.percentile(v, 75)
    feats[f"{prefix}_iqr"]      = np.percentile(v, 75) - np.percentile(v, 25)
    feats[f"{prefix}_velocity"]   = np.trapezoid(np.abs(v))
    feats[f"{prefix}_rms"]       = np.sqrt(np.mean(v**2))
    feats[f"{prefix}_zero_crossing"] = int(np.sum(np.diff(np.sign(v - np.mean(v))) != 0))
    peaks, _ = find_peaks(v)
    peak_vals = v[peaks] if len(peaks) > 0 else np.array([0.0])
    feats[f"{prefix}_n_peaks"]   = len(peaks)
    feats[f"{prefix}_peak_min"]  = float(np.min(peak_vals))
    feats[f"{prefix}_peak_max"]  = float(np.max(peak_vals))
    feats[f"{prefix}_peak_mean"] = float(np.mean(peak_vals))
    fft = np.abs(np.fft.rfft(v))
    feats[f"{prefix}_fft_sum"] = float(np.sum(fft))
    feats[f"{prefix}_energy"]  = float(np.sum(fft**2))
    return feats


def extract_features(df, prefix):
    feats = {}
    for axis in ["x", "y", "z"]:
        feats.update(axis_features(df[axis].values, f"{prefix}_{axis}"))

    x, y, z = df["x"].values, df["y"].values, df["z"].values
    mag = np.sqrt(x**2 + y**2 + z**2)
    feats[f"{prefix}_avg_magnitude"] = np.mean(mag)
    feats[f"{prefix}_cor_xy"]        = float(np.corrcoef(x, y)[0, 1])
    feats[f"{prefix}_cor_xz"]        = float(np.corrcoef(x, z)[0, 1])
    feats[f"{prefix}_cor_yz"]        = float(np.corrcoef(y, z)[0, 1])
    return feats


def cross_sensor_features(accel, gyro):
    feats = {}
    for axis in ["x", "y", "z"]:
        a = accel[axis].values
        g = gyro[axis].values
        n = min(len(a), len(g))
        feats[f"accel_gyro_cor_{axis}"] = float(np.corrcoef(a[:n], g[:n])[0, 1]) if n > 2 else 0.0
    return feats


def load_dataset(local_dir):
    data_path = Path(local_dir)
    user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"\nNajdených {len(user_dirs)} pouzivatelov")

    all_features, all_labels = [], []

    for user_dir in user_dirs:
        attempts = defaultdict(dict)
        for f in user_dir.glob("*.csv"):
            m = RE_ACCEL.match(f.name) or RE_GYRO.match(f.name)
            if m:
                key = "accel" if "accelerometer" in f.name.lower() else "gyro"
                attempts[int(m.group(1))][key] = f
        for attempt_num, files in sorted(attempts.items()):
            if "accel" not in files or "gyro" not in files:
                continue
            try:
                gyro  = parse_file(files["gyro"])
                gyro  = trim_gesture(gyro)
                accel = parse_file(files["accel"])
                accel = accel.iloc[:len(gyro)].reset_index(drop=True)
                if len(accel) < 10 or len(gyro) < 10:
                    continue
                feats = {}
                feats.update(extract_features(accel, "acc"))
                feats.update(extract_features(gyro,  "gyr"))
                feats.update(cross_sensor_features(accel, gyro))
                all_features.append(feats)
                all_labels.append(user_dir.name)
            except Exception as e:
                print(f"  [CHYBA] {user_dir.name} pokus {attempt_num}: {e}")

    df_feats = pd.DataFrame(all_features).fillna(0)
    print(f"Dataset: {len(df_feats)} vzoriek x {len(df_feats.columns)} priznakov")
    return df_feats.values.astype(np.float64), np.array(all_labels), df_feats.columns.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-dir", default=LOCAL_DATA_DIR)
    parser.add_argument("--fs", choices=["corr", "full"], default=None)
    parser.add_argument("--model", default="Random Forest",
                        choices=["SVM", "Random Forest", "XGBoost", "KNN"])
    args = parser.parse_args()

    if args.download:
        download_from_firebase(args.data_dir)

    X, y, feature_names = load_dataset(args.data_dir)
    X, feature_names = select_features(X, y, feature_names, gesture="table", model=args.model, mode=args.fs or "none")
    train_and_evaluate(X, y, feature_names, "gesture_model_stol.pkl")


if __name__ == "__main__":
    main()