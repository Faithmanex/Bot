import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory to store trained models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def extract_features_from_pivots(pivots_seq):
    vals = np.array([p["val"] for p in pivots_seq])
    idxs = np.array([p["idx"] for p in pivots_seq])

    min_val = vals.min()
    max_val = vals.max()
    val_range = max_val - min_val if max_val != min_val else 1e-6
    norm_vals = (vals - min_val) / val_range

    ratios = []
    for i in range(5):
        wave_current = abs(vals[i] - vals[i + 1])
        wave_next = abs(vals[i + 1] - vals[min(i + 2, 5)])
        ratios.append(wave_current / (wave_next + 1e-6))

    total_time = idxs[0] - idxs[5] if idxs[0] != idxs[5] else 1e-6
    time_diffs = []
    for i in range(5):
        time_diffs.append((idxs[i] - idxs[i + 1]) / total_time)

    features = np.concatenate([norm_vals, ratios, time_diffs])
    return features


def label_pivot_trade(df, occ_idx, entry_price, stop_loss, take_profit, is_buy):
    future_data = df.iloc[occ_idx + 1 :]
    if future_data.empty:
        return 0

    highs = future_data["High"].to_numpy()
    lows = future_data["Low"].to_numpy()

    if is_buy:
        sl_hits = lows <= stop_loss
        tp_hits = highs >= take_profit
    else:
        sl_hits = highs >= stop_loss
        tp_hits = lows <= take_profit

    first_sl = sl_hits.argmax() if sl_hits.any() else len(future_data)
    first_tp = tp_hits.argmax() if tp_hits.any() else len(future_data)

    if not sl_hits.any() and not tp_hits.any():
        return 0

    if first_tp < first_sl:
        return 1
    else:
        return 0


def get_all_pivot_sequences(df):
    pivots_df = df[(df["Is_High"].notna()) | (df["Is_Low"].notna())].copy()
    if len(pivots_df) < 6:
        return [], [], []

    sequences = []
    trigger_indices = []
    is_buy_list = []

    df_indices = {time_idx: i for i, time_idx in enumerate(df.index)}

    pivots_list = []
    for time_idx, row in pivots_df.iterrows():
        is_high = pd.notna(row["Is_High"])
        val = row["Is_High"] if is_high else row["Is_Low"]
        pivots_list.append({
            "val": float(val),
            "idx": df_indices[time_idx],
            "time": time_idx,
            "is_high": is_high
        })

    for i in range(len(pivots_list) - 5):
        seq = pivots_list[i : i + 6]
        seq.reverse()
        sequences.append(seq)
        trigger_indices.append(seq[0]["time"])
        is_buy_list.append(not seq[0]["is_high"])

    return sequences, trigger_indices, is_buy_list


def _build_xy(df, sequences, trigger_indices, is_buy_list, RR):
    X, y, meta = [], [], []
    for seq, trig_time, is_buy in zip(sequences, trigger_indices, is_buy_list):
        features = extract_features_from_pivots(seq)
        entry_price = seq[0]["val"]
        wave_size = abs(seq[1]["val"] - seq[2]["val"])
        if wave_size == 0:
            wave_size = 1e-4

        if is_buy:
            stop_loss = entry_price - (wave_size * 0.5)
            take_profit = entry_price + (entry_price - stop_loss) * RR
        else:
            stop_loss = entry_price + (wave_size * 0.5)
            take_profit = entry_price - (stop_loss - entry_price) * RR

        occ_idx = seq[0]["idx"]
        label = label_pivot_trade(df, occ_idx, entry_price, stop_loss, take_profit, is_buy)

        X.append(features)
        y.append(label)
        meta.append({"trigger_time": trig_time, "entry": entry_price, "sl": stop_loss, "tp": take_profit, "is_buy": is_buy})

    return np.array(X), np.array(y), meta


def _report_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name}:  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    unique, counts = np.unique(y_true, return_counts=True)
    print(f"  {name} class distribution: {dict(zip(unique, counts))}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def temporal_train_test_split(sequences, trigger_indices, is_buy_list, test_size=0.2):
    paired = list(zip(sequences, trigger_indices, is_buy_list))
    paired.sort(key=lambda x: x[1])

    n = len(paired)
    split_idx = int(n * (1 - test_size))

    train = paired[:split_idx]
    test = paired[split_idx:]

    if len(train) < 5 or len(test) < 5:
        print(f"[WARN] Temporal split too small: train={len(train)}, test={len(test)}. Adjust test_size.")
        return paired, [], [], [], [], []

    train_seq, train_tri, train_buy = zip(*train) if train else ([], [], [])
    test_seq, test_tri, test_buy = zip(*test) if test else ([], [], [])

    return list(train_seq), list(train_tri), list(train_buy), list(test_seq), list(test_tri), list(test_buy)


def _build_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )


def _metadata_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_pattern_meta.json")


def _save_training_metadata(symbol, cutoff_date, n_train, n_test):
    meta = {
        "train_cutoff": str(cutoff_date),
        "n_train": n_train,
        "n_test": n_test,
        "trained_at": str(pd.Timestamp.now()),
    }
    path = _metadata_path(symbol)
    with open(path, "w") as f:
        json.dump(meta, f)
    print(f"[INFO] Training metadata saved to {path}")


def _load_training_metadata(symbol):
    path = _metadata_path(symbol)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_saved_models():
    models = []
    if not os.path.isdir(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith("_pattern_meta.json"):
            symbol = fname.replace("_pattern_meta.json", "")
            meta_path = os.path.join(MODEL_DIR, fname)
            model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
            meta = None
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
            models.append({
                "symbol": symbol,
                "has_model": os.path.exists(model_path),
                "meta": meta,
            })
    return models


# Memory cache for loaded models
_loaded_models = {}


def predict_pattern_probability(symbol, pivots_seq):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
    if not os.path.exists(model_path):
        return 0.0

    try:
        if model_path not in _loaded_models:
            _loaded_models[model_path] = joblib.load(model_path)
        model = _loaded_models[model_path]
        features = extract_features_from_pivots(pivots_seq).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        return float(prob)
    except Exception as e:
        print(f"[ERROR] Inference error for {symbol}: {e}")
        return 0.0


def build_and_train_model(df, symbol, RR=5.0, test_size=0.2,
                          cutoff_date=None, retrain_on_all=True):
    """
    Trains a RandomForest with temporal isolation.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data with Is_High/Is_Low pivot columns and a DatetimeIndex.
    symbol : str
        Symbol name for model filename.
    RR : float
        Risk-to-reward ratio for label calculation.
    test_size : float
        Fraction of chronologically LAST sequences held out for testing (0.0 to 1.0).
        Ignored if cutoff_date is provided.
    cutoff_date : str or datetime, optional
        Explicit date cutoff. Sequences with trigger_time < cutoff_date go to train,
        >= cutoff_date go to test. Overrides test_size.
    retrain_on_all : bool
        If True, saves a final model trained on ALL data (for production).
        The validation metrics still come from the temporally-isolated test set.

    Returns
    -------
    dict with keys: success, train_metrics, test_metrics, n_train, n_test, cutoff
    """
    print(f"[INFO] Extracting pivot sequences for {symbol}...")
    sequences, trigger_indices, is_buy_list = get_all_pivot_sequences(df)

    if len(sequences) < 15:
        print(f"[WARN] Not enough pivot sequences ({len(sequences)}) for {symbol}. Need at least 15.")
        return {"success": False, "reason": f"Only {len(sequences)} sequences, need 15"}

    # Build full X, y
    X, y, meta = _build_xy(df, sequences, trigger_indices, is_buy_list, RR)

    # --- Date-based temporal split (no shuffle: first (1-test_size) of time range for training) ---
    if cutoff_date is None:
        sorted_times = sorted(trigger_indices)
        t_min = sorted_times[0]
        t_max = sorted_times[-1]
        cutoff_date = t_min + (1 - test_size) * (t_max - t_min)

    if isinstance(cutoff_date, str):
        cutoff_date = pd.Timestamp(cutoff_date)

    train_mask = np.array([t < cutoff_date for t in trigger_indices])
    test_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    used_cutoff = str(cutoff_date)

    if len(X_train) < 5 or len(X_test) < 5:
        print(f"[WARN] Temporal split too small: train={len(X_train)}, test={len(X_test)}. Adjust split.")
        return {"success": False, "reason": f"Train={len(X_train)}, Test={len(X_test)} — too small"}

    print(f"[INFO] Temporal split: {len(X_train)} train  |  {len(X_test)} test  |  cutoff={used_cutoff}")
    print(f"[INFO] Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[INFO] Test  class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- Train on training set only ---
    model = _build_model()
    model.fit(X_train, y_train)

    # --- Evaluate on test set (chronologically future) ---
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n[RESULTS] Temporal validation for {symbol}:")
    train_metrics = _report_metrics("Train (in-sample)",  y_train, y_train_pred, y_train_prob)
    test_metrics  = _report_metrics("Test  (out-of-sample)", y_test,  y_test_pred,  y_test_prob)

    # --- Train final model on ALL data and save ---
    if retrain_on_all:
        final = _build_model()
        final.fit(X, y)
        model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
        joblib.dump(final, model_path)
        print(f"[INFO] Final model (trained on all {len(X)} sequences) saved to {model_path}")
    else:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
        joblib.dump(model, model_path)
        print(f"[INFO] Model (trained on {len(X_train)} sequences only) saved to {model_path}")

    if model_path in _loaded_models:
        del _loaded_models[model_path]

    # Save training metadata for the temporal guard in MLPattern()
    _save_training_metadata(symbol, cutoff_date, int(train_mask.sum()), int(test_mask.sum()))

    result = {
        "success": True,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cutoff": used_cutoff,
        "test_size": test_size if cutoff_date is None else None
    }

    print(f"\n[SUMMARY] {symbol}: test accuracy={test_metrics['accuracy']:.4f}, "
          f"precision={test_metrics['precision']:.4f}, f1={test_metrics['f1']:.4f}")
    return result

