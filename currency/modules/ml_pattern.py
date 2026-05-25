import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# Directory to store trained models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def extract_features_from_pivots(pivots_seq):
    """
    Given a sequence of 6 consecutive pivot rows, extract normalized geometric features.
    pivots_seq: List of 6 dicts/tuples/rows containing 'val' (float price) and 'idx' (int location).
    """
    # Extract values and index locations
    vals = np.array([p["val"] for p in pivots_seq])
    idxs = np.array([p["idx"] for p in pivots_seq])

    # 1. Price normalization (Min-Max Scaling) relative to sequence range
    min_val = vals.min()
    max_val = vals.max()
    val_range = max_val - min_val if max_val != min_val else 1e-6
    norm_vals = (vals - min_val) / val_range

    # 2. Price swing ratios (consecutive wave lengths)
    ratios = []
    for i in range(5):
        wave_current = abs(vals[i] - vals[i + 1])
        wave_next = abs(vals[i + 1] - vals[min(i + 2, 5)])
        ratios.append(wave_current / (wave_next + 1e-6))

    # 3. Time difference normalization (spacing relative to total duration)
    total_time = idxs[0] - idxs[5] if idxs[0] != idxs[5] else 1e-6
    time_diffs = []
    for i in range(5):
        time_diffs.append((idxs[i] - idxs[i + 1]) / total_time)

    # Combine all features into a single 1D array (6 + 5 + 5 = 16 features)
    features = np.concatenate([norm_vals, ratios, time_diffs])
    return features


def label_pivot_trade(df, occ_idx, entry_price, stop_loss, take_profit, is_buy):
    """
    Labels a candidate trade by checking future price action.
    Returns 1 if take_profit is hit first, 0 if stop_loss is hit first, or None if neither.
    """
    future_data = df.iloc[occ_idx + 1 :]
    if future_data.empty:
        return 0

    highs = future_data["High"].to_numpy()
    lows = future_data["Low"].to_numpy()

    if is_buy:
        # Buy Setup: TP is high, SL is low
        sl_hits = lows <= stop_loss
        tp_hits = highs >= take_profit
    else:
        # Sell Setup: TP is low, SL is high
        sl_hits = highs >= stop_loss
        tp_hits = lows <= take_profit

    first_sl = sl_hits.argmax() if sl_hits.any() else len(future_data)
    first_tp = tp_hits.argmax() if tp_hits.any() else len(future_data)

    if not sl_hits.any() and not tp_hits.any():
        return 0  # didn't hit either: conservative label as 0

    if first_tp < first_sl:
        return 1  # TP hit first
    else:
        return 0  # SL hit first


def get_all_pivot_sequences(df):
    """
    Scans the dataframe to extract all valid 6-pivot sequences.
    Returns:
      sequences: List of lists, each containing 6 dicts: {'val': price, 'idx': int, 'is_high': bool}
      trigger_indices: List of pandas indices at the trigger point (most recent pivot in seq)
      is_buy_list: List of bool values indicating buy (True) or sell (False) setup
    """
    # Filter only rows that are pivots
    pivots_df = df[(df["Is_High"].notna()) | (df["Is_Low"].notna())].copy()
    if len(pivots_df) < 6:
        return [], [], []

    sequences = []
    trigger_indices = []
    is_buy_list = []

    # Get integer location of pivots in original df
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

    # Slide through pivot list to build 6-pivot sequences
    for i in range(len(pivots_list) - 5):
        # most recent is at i+5 (if we index linearly), or let's reverse order so index 0 is most recent
        seq = pivots_list[i : i + 6]
        seq.reverse()  # now seq[0] is most recent, seq[5] is oldest

        sequences.append(seq)
        trigger_indices.append(seq[0]["time"])
        # If the trigger pivot (most recent) is a Low, it's a candidate buy setup. Otherwise, sell setup.
        is_buy_list.append(not seq[0]["is_high"])

    return sequences, trigger_indices, is_buy_list


# Memory cache for loaded models
_loaded_models = {}


def predict_pattern_probability(symbol, pivots_seq):
    """
    Loads saved model (cached in memory) and predicts probability of a profitable setup.
    """
    model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
    if not os.path.exists(model_path):
        return 0.0

    try:
        if model_path not in _loaded_models:
            _loaded_models[model_path] = joblib.load(model_path)
        model = _loaded_models[model_path]
        features = extract_features_from_pivots(pivots_seq).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]  # probability of class 1
        return float(prob)
    except Exception as e:
        print(f"[ERROR] Inference error for {symbol}: {e}")
        return 0.0


def build_and_train_model(df, symbol, RR=5.0):
    """
    Builds features and target labels from historical data, trains a RandomForestClassifier, and saves it.
    """
    print(f"[INFO] Extracting features and training ML model for {symbol}...")
    sequences, trigger_indices, is_buy_list = get_all_pivot_sequences(df)
    
    if len(sequences) < 15:
        print(f"[WARN] Not enough pivot sequences ({len(sequences)}) to train ML model for {symbol}. Need at least 15.")
        return False

    X, y = [], []
    df_index_arr = df.index

    for seq, trig_time, is_buy in zip(sequences, trigger_indices, is_buy_list):
        features = extract_features_from_pivots(seq)
        
        # Calculate candidates entry, SL, TP
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

    X = np.array(X)
    y = np.array(y)

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"[INFO] Class distribution for {symbol}: {class_dist}")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
    joblib.dump(model, model_path)
    
    # Invalidate model cache to ensure the freshly trained model is used
    if model_path in _loaded_models:
        del _loaded_models[model_path]
        
    print(f"[INFO] ML model successfully saved to {model_path}")
    return True

