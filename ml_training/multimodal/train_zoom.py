#!/usr/bin/env python3
import re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis as sp_kurtosis
from core import (sanitize_path, parse_sensor_file, clip_sensor_to_segment,
                    sensor_paths_for, extract_sensor_features, cross_sensor_features,
                    train_and_evaluate)

FIREBASE_BUCKET = "dpapp-18ab8.firebasestorage.app"
GESTURE_PATH    = "touch_zoom_behametrics"
LOCAL_DATA_DIR  = str(Path(__file__).parent.parent.parent / "data" / "zoom")
SERVICE_ACCOUNT = "serviceAccountKey.json"

RE_TOUCH = re.compile(r"log(\d+)_touch\.csv",                 re.IGNORECASE)
RE_ACCEL = re.compile(r"log(\d+)_sensor_accelerometer\.csv",  re.IGNORECASE)
RE_GYRO  = re.compile(r"log(\d+)_sensor_gyroscope\.csv",      re.IGNORECASE)

TOUCH_COLS = ["type", "user_id", "timestamp_ns", "action", "action_detail",
              "pointer_id", "x", "y", "pressure", "size",
              "touch_major", "touch_minor", "raw_x", "raw_y"]

MIN_FRAMES      = 5
MIN_DIST_CHANGE = 50
GAP_NS          = 200_000_000
MARGIN_NS       = 50_000_000


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

    patterns = [RE_TOUCH, RE_ACCEL, RE_GYRO]
    for blob in blobs:
        parts = blob.name[len(GESTURE_PATH) + 1:].split("/", 1)
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

    print("Hotovo.")


def parse_touch_file(filepath):
    df = pd.read_csv(filepath, header=None, names=TOUCH_COLS)
    for col in ["timestamp_ns", "pointer_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["x", "y", "pressure", "size", "touch_major", "touch_minor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["timestamp_ns", "pointer_id", "x", "y"], inplace=True)
    df["pointer_id"] = df["pointer_id"].astype(int)
    df.sort_values("timestamp_ns", inplace=True)
    return df.reset_index(drop=True)


def segment_gestures(df):
    p1 = df[df["pointer_id"] == 1].sort_values("timestamp_ns")
    if p1.empty:
        return []

    seg_ranges = []
    seg_start  = p1.iloc[0]["timestamp_ns"]
    seg_end    = p1.iloc[0]["timestamp_ns"]

    for _, row in p1.iterrows():
        if row["timestamp_ns"] - seg_end > GAP_NS:
            seg_ranges.append((seg_start, seg_end))
            seg_start = row["timestamp_ns"]
        seg_end = row["timestamp_ns"]
    seg_ranges.append((seg_start, seg_end))

    segments = []
    for start_ts, end_ts in seg_ranges:
        seg = df[(df["timestamp_ns"] >= start_ts - MARGIN_NS) &
                 (df["timestamp_ns"] <= end_ts   + MARGIN_NS)].copy()
        if 0 not in seg["pointer_id"].values or 1 not in seg["pointer_id"].values:
            continue
        if len(seg) < MIN_FRAMES:
            continue
        segments.append(seg.reset_index(drop=True))

    return segments


def build_distance_series(df):
    moves = df[df["action"].isin(["move", "down", "pointer_down"])].copy()
    moves["time_ms"] = (moves["timestamp_ns"] - moves["timestamp_ns"].min()) / 1e6

    p0 = moves[moves["pointer_id"] == 0][["time_ms", "x", "y"]].rename(columns={"x": "x0", "y": "y0"})
    p1 = moves[moves["pointer_id"] == 1][["time_ms", "x", "y"]].rename(columns={"x": "x1", "y": "y1"})

    if p0.empty or p1.empty:
        return None, None

    p0 = p0.sort_values("time_ms").reset_index(drop=True)
    p1 = p1.sort_values("time_ms").reset_index(drop=True)
    merged = pd.merge_asof(p0, p1, on="time_ms", direction="nearest", tolerance=20.0)
    merged.dropna(inplace=True)

    if len(merged) < MIN_FRAMES:
        return None, None

    dist = np.sqrt((merged["x1"] - merged["x0"])**2 + (merged["y1"] - merged["y0"])**2)
    return merged["time_ms"].values, dist.values


def is_valid_zoom(times, dists):
    if times is None or len(dists) < MIN_FRAMES:
        return False
    return abs(dists[-1] - dists[0]) >= MIN_DIST_CHANGE


def extract_touch_features(times_ms, distances, seg_df=None):
    feats = {}
    d = distances
    t = times_ms
    duration = t[-1] - t[0] if len(t) > 1 else 0.0

    feats["dist_initial"]   = d[0]
    feats["dist_final"]     = d[-1]
    feats["dist_delta"]     = d[-1] - d[0]
    feats["zoom_factor"]    = d[-1] / d[0] if d[0] > 0 else 1.0
    feats["dist_min"]       = np.min(d)
    feats["dist_max"]       = np.max(d)
    feats["dist_mean"]      = np.mean(d)
    feats["dist_var"]       = np.var(d)
    feats["dist_std"]       = np.std(d)
    feats["dist_range"]     = np.max(d) - np.min(d)
    feats["dist_median"]    = np.median(d)
    feats["dist_q1"]        = np.percentile(d, 25)
    feats["dist_q3"]        = np.percentile(d, 75)
    feats["dist_iqr"]       = np.percentile(d, 75) - np.percentile(d, 25)
    feats["dist_skew"]      = float(skew(d))        if len(d) > 2 else 0.0
    feats["dist_kurt"]      = float(sp_kurtosis(d)) if len(d) > 3 else 0.0
    feats["duration_ms"]    = duration
    feats["zoom_direction"] = 1.0 if feats["dist_delta"] > 0 else -1.0

    if len(d) >= 4:
        feats["dist_at_25pct"] = d[len(d) // 4]
        feats["dist_at_50pct"] = d[len(d) // 2]
        feats["dist_at_75pct"] = d[3 * len(d) // 4]
    else:
        feats["dist_at_25pct"] = feats["dist_at_50pct"] = feats["dist_at_75pct"] = d[0]

    if len(d) > 1 and duration > 0:
        dt       = np.diff(t).clip(min=1e-3)
        velocity = np.diff(d) / dt
        feats["vel_mean"]            = np.mean(velocity)
        feats["vel_std"]             = np.std(velocity)
        feats["vel_max"]             = np.max(np.abs(velocity))
        mid = len(velocity) // 2
        feats["vel_first_half"]      = np.mean(velocity[:mid]) if mid > 0 else 0.0
        feats["vel_second_half"]     = np.mean(velocity[mid:]) if mid < len(velocity) else 0.0
        feats["n_direction_changes"] = float(np.sum(np.diff(np.sign(velocity)) != 0))
        if len(velocity) > 1:
            accel = np.diff(velocity)
            feats["accel_mean"] = np.mean(accel)
            feats["accel_std"]  = np.std(accel)
        else:
            feats["accel_mean"] = feats["accel_std"] = 0.0
    else:
        for k in ["vel_mean", "vel_std", "vel_max", "vel_first_half", "vel_second_half",
                  "n_direction_changes", "accel_mean", "accel_std"]:
            feats[k] = 0.0

    if seg_df is not None:
        moves = seg_df[seg_df["action"].isin(["move", "down", "pointer_down"])].copy()
        moves["time_ms"] = (moves["timestamp_ns"] - moves["timestamp_ns"].min()) / 1e6
        p0 = moves[moves["pointer_id"] == 0].sort_values("time_ms")
        p1 = moves[moves["pointer_id"] == 1].sort_values("time_ms")

        if not p0.empty and not p1.empty:
            cx_start = (p0["x"].iloc[0] + p1["x"].iloc[0]) / 2
            cy_start = (p0["y"].iloc[0] + p1["y"].iloc[0]) / 2
            cx_end   = (p0["x"].iloc[-1] + p1["x"].iloc[-1]) / 2
            cy_end   = (p0["y"].iloc[-1] + p1["y"].iloc[-1]) / 2
            feats["centroid_x_start"] = cx_start
            feats["centroid_y_start"] = cy_start
            feats["centroid_x_end"]   = cx_end
            feats["centroid_y_end"]   = cy_end
            feats["centroid_dx"]      = cx_end - cx_start
            feats["centroid_dy"]      = cy_end - cy_start
            feats["centroid_disp"]    = np.sqrt(feats["centroid_dx"]**2 + feats["centroid_dy"]**2)

            feats["angle_start"] = np.degrees(np.arctan2(
                p1["y"].iloc[0] - p0["y"].iloc[0], p1["x"].iloc[0] - p0["x"].iloc[0]))
            feats["angle_end"]   = np.degrees(np.arctan2(
                p1["y"].iloc[-1] - p0["y"].iloc[-1], p1["x"].iloc[-1] - p0["x"].iloc[-1]))
            feats["angle_delta"] = feats["angle_end"] - feats["angle_start"]

            def path_length(pts_x, pts_y):
                return float(np.sum(np.sqrt(np.diff(pts_x.values)**2 + np.diff(pts_y.values)**2)))

            feats["p0_path_length"] = path_length(p0["x"], p0["y"])
            feats["p1_path_length"] = path_length(p1["x"], p1["y"])
            feats["path_asymmetry"] = abs(feats["p0_path_length"] - feats["p1_path_length"])

            feats["p0_disp"] = np.sqrt((p0["x"].iloc[-1] - p0["x"].iloc[0])**2 +
                                       (p0["y"].iloc[-1] - p0["y"].iloc[0])**2)
            feats["p1_disp"] = np.sqrt((p1["x"].iloc[-1] - p1["x"].iloc[0])**2 +
                                       (p1["y"].iloc[-1] - p1["y"].iloc[0])**2)

            dur0 = p0["time_ms"].iloc[-1] - p0["time_ms"].iloc[0] + 1e-3
            dur1 = p1["time_ms"].iloc[-1] - p1["time_ms"].iloc[0] + 1e-3
            feats["p0_vel"]        = feats["p0_path_length"] / dur0
            feats["p1_vel"]        = feats["p1_path_length"] / dur1
            feats["vel_asymmetry"] = abs(feats["p0_vel"] - feats["p1_vel"])

            all_x = pd.concat([p0["x"], p1["x"]])
            all_y = pd.concat([p0["y"], p1["y"]])
            feats["bbox_width"]  = all_x.max() - all_x.min()
            feats["bbox_height"] = all_y.max() - all_y.min()

            feats["bbox_aspect"]   = feats["bbox_width"] / (feats["bbox_height"] + 1e-3)

            size = moves["size"].replace(0, np.nan).fillna(moves["size"].mean())
            feats["size_mean"] = float(size.mean())
            feats["size_std"]  = float(size.std()) if len(size) > 1 else 0.0

            major = moves["touch_major"].replace(0, np.nan).fillna(moves["touch_major"].mean())
            minor = moves["touch_minor"].replace(0, np.nan).fillna(moves["touch_minor"].mean())
            feats["touch_major_mean"] = float(major.mean())
            feats["touch_minor_mean"] = float(minor.mean())
            feats["aspect_ratio"]     = float(major.mean() / minor.mean()) if minor.mean() > 0 else 1.0
        else:
            for k in ["centroid_x_start", "centroid_y_start", "centroid_x_end", "centroid_y_end",
                      "centroid_dx", "centroid_dy", "centroid_disp",
                      "angle_start", "angle_end", "angle_delta",
                      "p0_path_length", "p1_path_length", "path_asymmetry",
                      "p0_disp", "p1_disp", "p0_vel", "p1_vel", "vel_asymmetry",
                      "bbox_width", "bbox_height", "bbox_aspect",
                      "size_mean", "size_std", "touch_major_mean", "touch_minor_mean",
                      "aspect_ratio"]:
                feats[k] = 0.0

    return feats


def load_dataset(local_dir):
    data_path = Path(local_dir)
    user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"\nNajdených {len(user_dirs)} pouzivatelov")

    all_features, all_labels = [], []
    n_with_acc, n_with_gyr = 0, 0

    for user_dir in user_dirs:
        for f in sorted(user_dir.glob("*.csv")):
            if not RE_TOUCH.match(f.name):
                continue
            try:
                df       = parse_touch_file(f)
                segments = segment_gestures(df)
                if not segments:
                    continue

                acc_path, gyr_path = sensor_paths_for(f)
                acc_full = parse_sensor_file(acc_path) if acc_path.exists() else None
                gyr_full = parse_sensor_file(gyr_path) if gyr_path.exists() else None

                for seg in segments:
                    times, dists = build_distance_series(seg)
                    if not is_valid_zoom(times, dists):
                        continue

                    feats = extract_touch_features(times, dists, seg_df=seg)

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
                    break
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
    train_and_evaluate(X, y, feature_names, "zoom_model_mm.pkl")


if __name__ == "__main__":
    main()
