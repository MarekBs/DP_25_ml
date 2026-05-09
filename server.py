import io
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks

app = Flask(__name__)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "gesture_model_zdvihnutie.pkl"
THRESHOLDS_PATH = ROOT / "optimal_thresholds.json"

model_data = joblib.load(MODEL_PATH)
with open(THRESHOLDS_PATH) as f:
    thresholds = json.load(f)

MODELS = model_data["models"]
FEATURE_NAMES = model_data["feature_names"]
MODEL_TYPE = model_data["model_type"]
THRESHOLD = thresholds["pickup"][MODEL_TYPE]["threshold"]

print(f"Model: {MODEL_TYPE}  |  threshold: {THRESHOLD:.4f}  |  users: {len(MODELS)}")


def parse_csv(content: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(content), header=None,
                     names=["sensor", "user_id", "timestamp_ns", "x", "y", "z"])
    df = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def trim_gesture(df, smooth=5, padding=10):
    mag = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    mag = mag.rolling(window=smooth, center=True, min_periods=1).mean()
    threshold = 0.15 * np.percentile(mag, 95)
    active = np.where(mag > threshold)[0]
    if len(active) == 0:
        return df
    start = max(0, active[0] - padding)
    end = min(len(df) - 1, active[-1] + padding)
    return df.iloc[start:end + 1].reset_index(drop=True)


def axis_features(v, prefix):
    feats = {}
    feats[f"{prefix}_min"] = np.min(v)
    feats[f"{prefix}_max"] = np.max(v)
    feats[f"{prefix}_mean"] = np.mean(v)
    feats[f"{prefix}_var"] = np.var(v)
    feats[f"{prefix}_std"] = np.std(v)
    feats[f"{prefix}_median"] = np.median(v)
    feats[f"{prefix}_skewness"] = skew(v)
    feats[f"{prefix}_kurtosis"] = sp_kurtosis(v)
    feats[f"{prefix}_q1"] = np.percentile(v, 25)
    feats[f"{prefix}_q3"] = np.percentile(v, 75)
    feats[f"{prefix}_iqr"] = np.percentile(v, 75) - np.percentile(v, 25)
    feats[f"{prefix}_velocity"] = float(np.trapezoid(np.abs(v)))
    feats[f"{prefix}_rms"] = np.sqrt(np.mean(v**2))
    feats[f"{prefix}_zero_crossing"] = int(np.sum(np.diff(np.sign(v - np.mean(v))) != 0))
    peaks, _ = find_peaks(v)
    peak_vals = v[peaks] if len(peaks) > 0 else np.array([0.0])
    feats[f"{prefix}_n_peaks"] = len(peaks)
    feats[f"{prefix}_peak_min"] = float(np.min(peak_vals))
    feats[f"{prefix}_peak_max"] = float(np.max(peak_vals))
    feats[f"{prefix}_peak_mean"] = float(np.mean(peak_vals))
    fft = np.abs(np.fft.rfft(v))
    feats[f"{prefix}_fft_sum"] = float(np.sum(fft))
    feats[f"{prefix}_energy"] = float(np.sum(fft**2))
    return feats


def extract_features(df, prefix):
    feats = {}
    for axis in ["x", "y", "z"]:
        feats.update(axis_features(df[axis].values, f"{prefix}_{axis}"))
    x, y, z = df["x"].values, df["y"].values, df["z"].values
    mag = np.sqrt(x**2 + y**2 + z**2)
    feats[f"{prefix}_avg_magnitude"] = np.mean(mag)
    feats[f"{prefix}_cor_xy"] = float(np.corrcoef(x, y)[0, 1])
    feats[f"{prefix}_cor_xz"] = float(np.corrcoef(x, z)[0, 1])
    feats[f"{prefix}_cor_yz"] = float(np.corrcoef(y, z)[0, 1])
    return feats


def cross_sensor_features(accel, gyro):
    feats = {}
    for axis in ["x", "y", "z"]:
        a = accel[axis].values
        g = gyro[axis].values
        n = min(len(a), len(g))
        feats[f"accel_gyro_cor_{axis}"] = float(np.corrcoef(a[:n], g[:n])[0, 1]) if n > 2 else 0.0
    return feats


def build_feature_vector(accel_csv: str, gyro_csv: str) -> np.ndarray:
    accel = parse_csv(accel_csv)
    gyro = parse_csv(gyro_csv)

    gyro = trim_gesture(gyro)
    accel = accel.iloc[:len(gyro)].reset_index(drop=True)

    if len(accel) < 10 or len(gyro) < 10:
        raise ValueError(f"Príliš málo vzoriek: accel={len(accel)}, gyro={len(gyro)}")

    all_feats = {}
    all_feats.update(extract_features(accel, "acc"))
    all_feats.update(extract_features(gyro, "gyr"))
    all_feats.update(cross_sensor_features(accel, gyro))

    row = [all_feats.get(f, 0.0) for f in FEATURE_NAMES]
    return np.array([row], dtype=np.float64)


@app.route("/verify/zdvihnutie", methods=["POST"])
def verify_zdvihnutie():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    user_id = body.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if user_id not in MODELS:
        return jsonify({"error": f"Neznámy používateľ: {user_id}",
                        "known_users": sorted(MODELS.keys())}), 404

    accel_csv = body.get("accel_csv", "")
    gyro_csv = body.get("gyro_csv", "")
    if not accel_csv or not gyro_csv:
        return jsonify({"error": "accel_csv and gyro_csv required"}), 400

    try:
        X = build_feature_vector(accel_csv, gyro_csv)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Chyba pri extrakcii príznakov: {e}"}), 500

    confidence = float(MODELS[user_id].predict_proba(X)[0, 1])
    authenticated = confidence >= THRESHOLD

    return jsonify({
        "authenticated": authenticated,
        "confidence": round(confidence, 4),
        "threshold": round(THRESHOLD, 4),
        "model_type": MODEL_TYPE,
        "user_id": user_id,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
