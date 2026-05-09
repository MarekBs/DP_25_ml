#!/usr/bin/env python3
import re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks
from core import (sanitize_path, parse_sensor_file, clip_sensor_to_segment,
                    sensor_paths_for, extract_sensor_features, cross_sensor_features,
                    train_and_evaluate)

FIREBASE_BUCKET = "dpapp-18ab8.firebasestorage.app"
GESTURE_PATH    = "touch_gallery_behametrics"
DIRECTIONS      = ["doprava", "dolava"]
LOCAL_DATA_DIR  = str(Path(__file__).parent.parent.parent / "data" / "swipe")
SERVICE_ACCOUNT = "serviceAccountKey.json"

RE_TOUCH = re.compile(r"kolo(\d+)_(doprava|dolava)_touch\.csv",                re.IGNORECASE)
RE_ACCEL = re.compile(r"kolo(\d+)_(doprava|dolava)_sensor_accelerometer\.csv", re.IGNORECASE)
RE_GYRO  = re.compile(r"kolo(\d+)_(doprava|dolava)_sensor_gyroscope\.csv",     re.IGNORECASE)

TOUCH_COLS = ["type", "user_id", "timestamp_ns", "action", "action_detail",
              "pointer_id", "x", "y", "pressure", "size",
              "touch_major", "touch_minor", "raw_x", "raw_y"]

MIN_POINTS   = 5
MIN_DISTANCE = 150


def download_from_firebase(local_dir):
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage

    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            credentials.Certificate(SERVICE_ACCOUNT),
            {"storageBucket": FIREBASE_BUCKET}
        )

    bucket   = fb_storage.bucket()
    patterns = [RE_TOUCH, RE_ACCEL, RE_GYRO]
    total    = 0
    for direction in DIRECTIONS:
        prefix = f"{GESTURE_PATH}/{direction}/"
        blobs  = list(bucket.list_blobs(prefix=prefix))
        print(f"Firebase [{direction}]: najdených {len(blobs)} suborov")

        for blob in blobs:
            parts = blob.name[len(prefix):].split("/", 1)
            if len(parts) != 2 or not parts[1].endswith(".csv"):
                continue
            user_id, filename = parts
            if not any(p.match(filename) for p in patterns):
                continue
            safe_user_id = sanitize_path(user_id)
            dest = Path(local_dir) / safe_user_id / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                print(f"  Stahujem: {user_id}/{filename}")
                blob.download_to_filename(str(dest))
                total += 1

    print(f"Hotovo. Stiahnutých {total} nových suborov.")


def parse_touch_file(filepath):
    df = pd.read_csv(filepath, header=None, names=TOUCH_COLS)
    for col in ["timestamp_ns", "pointer_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["x", "y", "pressure", "size", "touch_major", "touch_minor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["timestamp_ns", "x", "y"], inplace=True)
    df = df[df["pointer_id"] == 0].copy()
    df.sort_values("timestamp_ns", inplace=True)
    return df.reset_index(drop=True)


def segment_gestures(df, expected_direction):
    segments  = []
    start_idx = None
    for i, row in df.iterrows():
        if row["action"] == "down":
            start_idx = i
        elif row["action"] == "up" and start_idx is not None:
            seg = df.loc[start_idx:i].copy()
            if seg.iloc[-1]["x"] == 0:
                seg = seg.iloc[:-1]
            seg = seg.reset_index(drop=True)
            if len(seg) >= MIN_POINTS:
                delta_x = seg["x"].iloc[-1] - seg["x"].iloc[0]
                dist    = np.sqrt(delta_x**2 + (seg["y"].iloc[-1] - seg["y"].iloc[0])**2)
                correct_dir = (expected_direction == "doprava" and delta_x < 0) or \
                              (expected_direction == "dolava"  and delta_x > 0)
                if dist >= MIN_DISTANCE and correct_dir:
                    segments.append(seg)
            start_idx = None
    return segments


def touch_axis_features(v, t, prefix):
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

    dt = np.diff(t).clip(min=1e-3)
    feats[f"{prefix}_velocity"]      = np.mean(np.abs(np.diff(v) / dt)) if len(v) > 1 else 0.0
    feats[f"{prefix}_rms"]           = np.sqrt(np.mean(v**2))
    feats[f"{prefix}_zero_crossing"] = int(np.sum(np.diff(np.sign(v - np.mean(v))) != 0))

    peaks, _ = find_peaks(v)
    pv = v[peaks] if len(peaks) > 0 else np.array([0.0])
    feats[f"{prefix}_peak_avg_distance"] = float(np.mean(np.diff(peaks))) if len(peaks) > 1 else 0.0
    feats[f"{prefix}_peak_min"]  = float(np.min(pv))
    feats[f"{prefix}_peak_max"]  = float(np.max(pv))
    feats[f"{prefix}_peak_mean"] = float(np.mean(pv))

    fft = np.abs(np.fft.rfft(v))
    feats[f"{prefix}_fft_sum"] = float(np.sum(fft))
    feats[f"{prefix}_energy"]  = float(np.sum(fft**2))

    return feats


def extract_touch_features(df, direction_label, screen_w=1080.0, screen_h=2340.0):
    feats = {}
    x = df["x"].values / screen_w
    y = df["y"].values / screen_h
    t = (df["timestamp_ns"].values - df["timestamp_ns"].values[0]) / 1e6

    feats.update(touch_axis_features(x, t, "x"))
    feats.update(touch_axis_features(y, t, "y"))

    feats["cor_xy"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0

    dx          = np.diff(x)
    dy          = np.diff(y)
    steps       = np.sqrt(dx**2 + dy**2)
    total_path  = np.sum(steps)
    direct_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

    feats["duration_ms"]  = t[-1] - t[0] if len(t) > 1 else 0.0
    feats["total_path"]   = total_path
    feats["straightness"] = direct_dist / total_path if total_path > 0 else 1.0
    feats["direct_dist"]  = direct_dist
    feats["start_x"]      = x[0]
    feats["start_y"]      = y[0]
    feats["end_x"]        = x[-1]
    feats["end_y"]        = y[-1]

    size = df["size"].replace(0, np.nan).fillna(df["size"].mean()).values
    feats["size_mean"] = float(np.mean(size))
    feats["size_std"]  = float(np.std(size))

    major = df["touch_major"].replace(0, np.nan).fillna(df["touch_major"].mean()).values
    minor = df["touch_minor"].replace(0, np.nan).fillna(df["touch_minor"].mean()).values
    feats["touch_major_mean"] = float(np.mean(major))
    feats["touch_minor_mean"] = float(np.mean(minor))
    feats["aspect_ratio"]     = float(np.mean(major) / np.mean(minor)) if np.mean(minor) > 0 else 1.0

    feats["swipe_direction"] = 1.0 if direction_label == "doprava" else -1.0

    return feats


def estimate_screen_size(user_dir):
    all_x, all_y = [], []
    for f in user_dir.glob("*.csv"):
        if not RE_TOUCH.match(f.name):
            continue
        try:
            df = parse_touch_file(f)
            df = df[df["action"] != "up"]
            all_x.extend(df["x"].dropna().tolist())
            all_y.extend(df["y"].dropna().tolist())
        except:
            pass
    if not all_x:
        return 1080.0, 2340.0
    return max(all_x), max(all_y)


def load_dataset(local_dir):
    data_path = Path(local_dir)
    user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"\nNajdených {len(user_dirs)} pouzivatelov")

    all_features, all_labels = [], []
    n_with_acc, n_with_gyr = 0, 0

    for user_dir in user_dirs:
        screen_w, screen_h = estimate_screen_size(user_dir)
        for f in sorted(user_dir.glob("*.csv")):
            m = RE_TOUCH.match(f.name)
            if not m:
                continue
            direction = m.group(2).lower()
            try:
                df       = parse_touch_file(f)
                segments = segment_gestures(df, direction)
                if not segments:
                    continue

                acc_path, gyr_path = sensor_paths_for(f)
                acc_full = parse_sensor_file(acc_path) if acc_path.exists() else None
                gyr_full = parse_sensor_file(gyr_path) if gyr_path.exists() else None

                for seg in segments:
                    feats = extract_touch_features(seg, direction, screen_w, screen_h)

                    seg_start = int(seg["timestamp_ns"].min())
                    seg_end   = int(seg["timestamp_ns"].max())
                    acc_clip  = clip_sensor_to_segment(acc_full, seg_start, seg_end)
                    gyr_clip  = clip_sensor_to_segment(gyr_full, seg_start, seg_end)
                    if acc_clip is not None: n_with_acc += 1
                    if gyr_clip is not None: n_with_gyr += 1

                    feats.update(extract_sensor_features(acc_clip, "acc"))
                    feats.update(extract_sensor_features(gyr_clip, "gyr"))
                    feats.update(cross_sensor_features(acc_clip, gyr_clip))

                    all_features.append(feats)
                    all_labels.append(user_dir.name)
            except Exception as e:
                print(f"  [CHYBA] {user_dir.name} / {f.name}: {e}")

    df_feats = pd.DataFrame(all_features).fillna(0)
    print(f"Dataset: {len(df_feats)} vzoriek x {len(df_feats.columns)} priznakov")
    print(f"  segmentov s accel: {n_with_acc}, s gyro: {n_with_gyr}")
    return df_feats.values.astype(np.float64), np.array(all_labels), df_feats.columns.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-dir", default=LOCAL_DATA_DIR)
    args = parser.parse_args()

    if args.download:
        download_from_firebase(args.data_dir)

    X, y, feature_names = load_dataset(args.data_dir)
    train_and_evaluate(X, y, feature_names, "swipe_model_mm.pkl")


if __name__ == "__main__":
    main()
