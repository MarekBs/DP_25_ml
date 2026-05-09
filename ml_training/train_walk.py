import re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks
from training import train_and_evaluate
from feature_selection import select_features

LOCAL_DATA_DIR = str(Path(__file__).parent.parent / "data" / "walk")

WINDOW_SIZE = 256   # vzorky na okno
WINDOW_STEP = 128   # krok (50% overlap)


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
    feats[f"{prefix}_velocity"]      = np.trapezoid(np.abs(v))
    feats[f"{prefix}_rms"]           = np.sqrt(np.mean(v**2))
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

def extract_sensor_features(df_xyz, prefix):
    feats = {}
    for axis in ["x", "y", "z"]:
        feats.update(axis_features(df_xyz[axis].values, f"{prefix}_{axis}"))
    x, y, z = df_xyz["x"].values, df_xyz["y"].values, df_xyz["z"].values
    mag = np.sqrt(x**2 + y**2 + z**2)
    feats[f"{prefix}_avg_magnitude"] = np.mean(mag)
    feats[f"{prefix}_cor_xy"]        = float(np.corrcoef(x, y)[0, 1])
    feats[f"{prefix}_cor_xz"]        = float(np.corrcoef(x, z)[0, 1])
    feats[f"{prefix}_cor_yz"]        = float(np.corrcoef(y, z)[0, 1])
    return feats

def cross_sensor_features(acc_df, gyr_df):
    feats = {}
    for axis in ["x", "y", "z"]:
        a = acc_df[axis].values
        g = gyr_df[axis].values
        n = min(len(a), len(g))
        feats[f"accel_gyro_cor_{axis}"] = float(np.corrcoef(a[:n], g[:n])[0, 1]) if n > 2 else 0.0
    return feats

def extract_window_features(window_df):
    acc_df = window_df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]]
    acc_df = acc_df.rename(columns=lambda c: c.split(".")[-1])
    gyr_df = window_df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]]
    gyr_df = gyr_df.rename(columns=lambda c: c.split(".")[-1])

    feats = {}
    feats.update(extract_sensor_features(acc_df, "acc"))
    feats.update(extract_sensor_features(gyr_df, "gyr"))
    feats.update(cross_sensor_features(acc_df, gyr_df))
    for axis in ["x", "y", "z"]:
        feats[f"grav_{axis}_mean"] = window_df[f"gravity.{axis}"].mean()
    return feats

def load_dataset(local_dir, window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
    data_path = Path(local_dir)
    csv_files = sorted(data_path.glob("sub_*.csv"))
    print(f"\nNajdených {len(csv_files)} subjektov")

    all_features, all_labels = [], []

    for csv_file in csv_files:
        subject_id = csv_file.stem
        try:
            df = pd.read_csv(csv_file, sep=None, engine="python")
            df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c.startswith("index")], errors="ignore")
            df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

            n_windows = 0
            for start in range(0, len(df) - window_size + 1, window_step):
                window = df.iloc[start : start + window_size]
                feats  = extract_window_features(window)
                all_features.append(feats)
                all_labels.append(subject_id)
                n_windows += 1

            print(f"  {subject_id}: {len(df)} vzoriek → {n_windows} okien")

        except Exception as e:
            print(f"  [CHYBA] {subject_id}: {e}")

    df_feats = pd.DataFrame(all_features).fillna(0)
    print(f"\nDataset: {len(df_feats)} vzoriek x {len(df_feats.columns)} príznakov")
    return df_feats.values.astype(np.float64), np.array(all_labels), df_feats.columns.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default=LOCAL_DATA_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--window-step", type=int, default=WINDOW_STEP)
    args = parser.parse_args()

    X, y, feature_names = load_dataset(args.data_dir, args.window_size, args.window_step)
    X, feature_names = select_features(X, y, feature_names)
    train_and_evaluate(X, y, feature_names, "walk_model.pkl", min_samples=5)

if __name__ == "__main__":
    main()