import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

from .modules.strategy import Strategy
from .modules.ml_pattern import build_and_train_model, MODEL_DIR
from .settings import load_settings, HISTORY_DATA_DIR, BACKTEST_SUMMARY_DIR as _BSR
from .unified_trading import prep_data, clean_data, detect_pivot_points

# Load configurations
settings = load_settings()


def run_sweep(symbol="EURUSD", timeframe_name="M5"):
    """
    Performs a rapid grid sweep over ML confidence thresholds and Risk-to-Reward ratios
    to find the absolute best pattern parameters.
    """
    print("=" * 70)
    print(f"   STARTING HYPER-SWEEP TO FIND BEST ML PATTERN SETTINGS FOR {symbol}   ")
    print("=" * 70)

    # Initialize MT5 to fetch data if necessary
    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed.")
        return

    # Check if history CSV exists
    filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
    if not os.path.exists(filename):
        print(f"[INFO] History file not found. Fetching historical data from MT5...")
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M10": mt5.TIMEFRAME_M10,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        timeframe = tf_map.get(timeframe_name, mt5.TIMEFRAME_M5)
        start = datetime(2025, 1, 1)
        end = datetime.now()
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None or len(rates) == 0:
            print(f"[ERROR] Could not fetch data from MT5 for {symbol}. Verify symbol is in Market Watch.")
            mt5.shutdown()
            return
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        rename_map = {
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "tick_volume": "Volume", "real_volume": "Volume"
        }
        df.rename(columns=rename_map, inplace=True)
        os.makedirs(HISTORY_DATA_DIR, exist_ok=True)
        df.to_csv(filename, index=False)

    mt5.shutdown()

    # Load and clean data
    df = prep_data(symbol, timeframe_name)
    clean_data(df, symbol)
    detect_pivot_points(df, symbol)

    # Make sure we have a trained model
    model_path = os.path.join(MODEL_DIR, f"{symbol}_pattern_model.joblib")
    if not os.path.exists(model_path):
        print(f"[INFO] Saved model for {symbol} not found. Auto-training now...")
        build_and_train_model(df, symbol, RR=5.0)

    # Instantiate Strategy with symbol
    strategy = Strategy(df, symbol=symbol)

    # Define sweep grid
    rr_values = [2.0, 3.0, 4.0, 5.0, 6.0]
    confidence_thresholds = [0.50, 0.54, 0.58, 0.60, 0.62, 0.65]

    results = []

    # Get sequences to check trade triggers
    from .modules.ml_pattern import get_all_pivot_sequences, predict_pattern_probability
    sequences, trigger_indices, is_buy_list = get_all_pivot_sequences(strategy.new_df)

    if not sequences:
        print("[ERROR] No pivot sequences detected. Cannot optimize.")
        return

    print(f"\nScanning {len(sequences)} historical swings across {len(rr_values) * len(confidence_thresholds)} combinations...")
    t0 = time.time()

    # Pre-calculate prediction probabilities for all sequences to speed up calculations
    probs = [predict_pattern_probability(symbol, seq) for seq in sequences]

    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    index_arr = df.index

    for rr in rr_values:
        for threshold in confidence_thresholds:
            # Simulate trading results
            initial_balance = 1000.0
            balance = initial_balance
            risk_amount = 25.0
            wins, losses, neither = 0, 0, 0
            trades_taken = 0

            for seq, trig_time, is_buy, prob in zip(sequences, trigger_indices, is_buy_list, probs):
                if prob < threshold:
                    continue

                trades_taken += 1
                entry_price = seq[0]["val"]
                wave_size = abs(seq[1]["val"] - seq[2]["val"])
                if wave_size == 0:
                    wave_size = 1e-4

                if is_buy:
                    stop_loss = entry_price - (wave_size * 0.5)
                    take_profit = entry_price + (entry_price - stop_loss) * rr
                else:
                    stop_loss = entry_price + (wave_size * 0.5)
                    take_profit = entry_price - (stop_loss - entry_price) * rr

                try:
                    occ_loc = index_arr.get_loc(trig_time)
                except KeyError:
                    neither += 1
                    continue

                # Check outcome
                future_high = high_arr[occ_loc + 1:]
                future_low = low_arr[occ_loc + 1:]

                entry_mask = future_high >= entry_price
                if not entry_mask.any():
                    neither += 1
                    continue

                entry_pos = entry_mask.argmax()
                post_high = future_high[entry_pos:]
                post_low = future_low[entry_pos:]

                sl_mask = post_high >= stop_loss if not is_buy else post_low <= stop_loss
                tp_mask = post_low <= take_profit if not is_buy else post_high >= take_profit

                sl_pos = sl_mask.argmax() if sl_mask.any() else len(post_high)
                tp_pos = tp_mask.argmax() if tp_mask.any() else len(post_low)

                if sl_mask.any() and sl_pos <= tp_pos:
                    balance -= risk_amount
                    losses += 1
                elif tp_mask.any():
                    balance += risk_amount * rr
                    wins += 1
                else:
                    neither += 1

            total_closed = wins + losses
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
            net_profit = balance - initial_balance

            results.append({
                "RR": rr,
                "Threshold": threshold,
                "Trades": trades_taken,
                "Wins": wins,
                "Losses": losses,
                "WinRate": win_rate,
                "NetProfit": net_profit,
                "FinalBalance": balance
            })

    sweep_df = pd.DataFrame(results)
    
    # Filter to require at least 5 trades to ensure statistical significance
    filtered_df = sweep_df[sweep_df["Trades"] >= 5].copy()
    if filtered_df.empty:
        filtered_df = sweep_df.copy()

    # Sort by Net Profit descending
    sorted_df = filtered_df.sort_values(by="NetProfit", ascending=False).reset_index(drop=True)
    t_duration = time.time() - t0

    # Print beautiful leaderboard
    print("\n" + "=" * 80)
    print(f"                     *** ML PATTERN SWEEP LEADERBOARD (Top 10) ***             ")
    print("=" * 80)
    print(f"{'Rank':<6}{'Risk-to-Reward (RR)':<22}{'Confidence Thr':<18}{'Total Trades':<14}{'Win Rate':<12}{'Net Profit':<12}")
    print("-" * 80)
    
    for i in range(min(10, len(sorted_df))):
        r = sorted_df.iloc[i]
        rank_str = f"#{i+1}"
        rr_str = f"{r['RR']:.1f}:1"
        thresh_str = f"{r['Threshold']*100:.0f}%"
        trades_str = f"{int(r['Trades'])}"
        wr_str = f"{r['WinRate']:.1f}%"
        profit_str = f"${r['NetProfit']:+,.2f}"
        
        # Highlight rank #1
        if i == 0:
            print(f"\033[92m{rank_str:<6}{rr_str:<22}{thresh_str:<18}{trades_str:<14}{wr_str:<12}{profit_str:<12}\033[0m")
        else:
            print(f"{rank_str:<6}{rr_str:<22}{thresh_str:<18}{trades_str:<14}{wr_str:<12}{profit_str:<12}")

    print("=" * 80)
    print(f"Sweep completed in {t_duration:.2f} seconds.")
    print("=" * 80)

    top10 = sorted_df.head(10).to_dict(orient="records")

    # Persist top-10 so the GUI can offer a selection dialog
    os.makedirs(_BSR, exist_ok=True)
    top10_path = os.path.join(_BSR, f"sweep_top10_{symbol}.json")
    with open(top10_path, "w") as fh:
        json.dump(top10, fh, indent=4)

    if top10:
        best = top10[0]
        print(f"\n[TOP RESULT]: RR {best['RR']:.1f}:1 | Threshold {best['Threshold']*100:.0f}% | "
              f"Win Rate {best['WinRate']:.1f}% | Net Profit ${best['NetProfit']:+,.2f}")

    print(f"\n[INFO] Top-10 results saved — select your preferred setup in the GUI to save it.")
    print("=" * 80 + "\n")


def save_sweep_result(symbol, result):
    """
    Persist a chosen sweep result dict to currency/settings.json.
    result must contain keys: RR, Threshold.
    """
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r") as fh:
                current_settings = json.load(fh)
        else:
            current_settings = {}

        if symbol not in current_settings:
            current_settings[symbol] = {}

        current_settings[symbol]["best_rr"] = float(result["RR"])
        current_settings[symbol]["best_threshold"] = float(result["Threshold"])

        with open(settings_path, "w") as fh:
            json.dump(current_settings, fh, indent=4)
        return True, f"RR={result['RR']:.1f}, Threshold={result['Threshold']*100:.0f}%"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    import sys
    symbol_arg = "EURUSD"
    if len(sys.argv) > 1:
        symbol_arg = sys.argv[1]
    run_sweep(symbol=symbol_arg)
