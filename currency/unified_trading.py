import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
from datetime import datetime, timedelta
import os
import time

from .modules.strategy import Strategy
from .modules import trading_pairs
from .settings import load_settings, HISTORY_DATA_DIR, BACKTEST_SUMMARY_DIR
from .modules.lot_size import get_lot_size

# Load global settings
settings = load_settings()

timeframes = {
    "M5": mt5.TIMEFRAME_M5,
}
start_time = datetime(2024, 1, 1, 0, 0, 0)
end_time = datetime.now()


def initialize_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code = {mt5.last_error()}")
        return False
    return True


def shutdown_mt5():
    mt5.shutdown()


def get_historical_data(symbol, timeframe, timeframe_name, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol}, error code = {mt5.last_error()}")
        return False
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # Rename columns to capitalized casing expected by strategies and backtester
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
        "real_volume": "Volume"
    }
    df.rename(columns=rename_map, inplace=True)
    
    filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
    df.to_csv(filename, index=False)
    return True


def prep_data(symbol, timeframe_name, visualize=False):
    filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    ohlcv_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if visualize:
        mpf.plot(ohlcv_df, type="candle", style="line", title=f"{symbol} {timeframe_name}", volume=True)
    return ohlcv_df


def clean_data(df, symbol, visualize=False):
    sym_cfg = settings.get(symbol, {"polyorder": 2, "window_length": 5})
    close_prices = df["Close"].to_numpy()
    smoothed_close = savgol_filter(close_prices, sym_cfg["window_length"], sym_cfg["polyorder"])
    df["smoothed_close"] = smoothed_close
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, close_prices, label="Close Price")
        plt.plot(df.index, smoothed_close, label="Smoothed Close Price")
        plt.legend()
        plt.show()


def detect_pivot_points(df, symbol, visualize=False):
    order = settings.get(symbol, {"order": 5})["order"]
    smoothed_close = df["smoothed_close"].to_numpy()
    highs = argrelextrema(smoothed_close, np.greater, mode="wrap", order=order)[0]
    lows = argrelextrema(smoothed_close, np.less, mode="wrap", order=order)[0]
    df.loc[df.index[highs], "Is_High"] = df["High"].iloc[highs]
    df.loc[df.index[lows], "Is_Low"] = df["Low"].iloc[lows]
    if visualize:
        apd = [
            mpf.make_addplot(df["Is_High"], scatter=True, markersize=30, marker="^", color="g"),
            mpf.make_addplot(df["Is_Low"], scatter=True, markersize=30, marker="v", color="r"),
        ]
        mpf.plot(df, type="candle", addplot=apd, style="charles", title=f"{symbol} Pivots")


def _get_pending_entry_prices(symbol):
    """Return a set of entry prices for all currently pending orders on this symbol."""
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        return set()
    return {o.price_open for o in orders}


def run_strategy(df, plot_df, RR, initial_balance, risk_amount, risk_type, symbol, live_trading=False):
    balance = mt5.account_info().balance if live_trading else initial_balance
    wins, losses, neither = 0, 0, 0
    results = []
    balance_history = []

    pending_prices = _get_pending_entry_prices(symbol) if live_trading else set()

    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    index_arr = df.index

    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence

        if live_trading and entry_price in pending_prices:
            continue

        try:
            occ_loc = index_arr.get_loc(occurrence_time)
        except KeyError:
            neither += 1
            continue

        # Vectorized: search for entry, SL, TP after occurrence
        future_high = high_arr[occ_loc + 1:]
        future_low = low_arr[occ_loc + 1:]
        future_index = index_arr[occ_loc + 1:]

        entry_mask = future_high >= entry_price
        if not entry_mask.any():
            neither += 1
            if live_trading:
                _place_pending_order(symbol, entry_price, stop_loss, take_profit,
                                     risk_amount, risk_type, balance)
                pending_prices.add(entry_price)
            results.append({
                "Occurrence": occurrence_time, "Entry": entry_price,
                "Stop_Loss": stop_loss, "Take_Profit": take_profit,
                "Result": "Pending", "Balance": balance,
            })
            balance_history.append({"Occurrence": occurrence_time, "Balance": balance})
            continue

        entry_pos = entry_mask.argmax()
        post_high = future_high[entry_pos:]
        post_low = future_low[entry_pos:]

        sl_mask = post_high >= stop_loss
        tp_mask = post_low <= take_profit

        sl_pos = sl_mask.argmax() if sl_mask.any() else len(post_high)
        tp_pos = tp_mask.argmax() if tp_mask.any() else len(post_low)

        if sl_mask.any() and sl_pos <= tp_pos:
            balance -= risk_amount
            result = "SL"
            losses += 1
        elif tp_mask.any():
            balance += risk_amount * RR
            result = "TP"
            wins += 1
        else:
            result = "Pending"
            neither += 1

        results.append({
            "Occurrence": occurrence_time, "Entry": entry_price,
            "Stop_Loss": stop_loss, "Take_Profit": take_profit,
            "Result": result, "Balance": balance,
        })
        balance_history.append({"Occurrence": occurrence_time, "Balance": balance})

    return pd.DataFrame(results), wins, losses, neither


def _place_pending_order(symbol, entry_price, stop_loss, take_profit, risk_amount, risk_type, balance):
    volume = get_lot_size(
        risk_amount=risk_amount,
        stop_loss=stop_loss,
        account_currency="USD",
        symbol=symbol,
        risk_type=risk_type,
        account_balance=balance,
        entry_price=entry_price,
    )
    if volume is None:
        print(f"Could not calculate lot size for {symbol} @ {entry_price}. Skipping.")
        return

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL_LIMIT,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,
        "magic": 0,
        "comment": "Echelnet Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to place order for {symbol} @ {entry_price}: retcode={res.retcode}")


def analyze_symbol(symbol, live_trading=False, config=None):
    if config is None:
        config = {}

    summary_results = []
    strategies = [config.get("strategy", "Noir")]
    timeframe_name = config.get("timeframe", "M5")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    timeframe = tf_map.get(timeframe_name, mt5.TIMEFRAME_M5)

    start = config.get("start_time", start_time)
    end = config.get("end_time", end_time)

    if not get_historical_data(symbol, timeframe, timeframe_name, start, end):
        print(f"Skipping {symbol} {timeframe_name} — no data available.")
        return

    df = prep_data(symbol, timeframe_name)
    clean_data(df, symbol)
    detect_pivot_points(df, symbol)

    for strategy_name in strategies:
        strategy = Strategy(df)
        rr = config.get("rr", 5.0)
        plot_df = getattr(strategy, strategy_name)(RR=rr)

        initial_balance = config.get("initial_balance", 1000.0)
        risk_amount = config.get("risk_amount", 25.0)
        risk_type = config.get("risk_type", "fixed")

        backtest_results_df, wins, losses, neither = run_strategy(
            df, plot_df, RR=rr, initial_balance=initial_balance,
            risk_amount=risk_amount, risk_type=risk_type,
            symbol=symbol, live_trading=live_trading,
        )

        if not backtest_results_df.empty:
            final_balance = backtest_results_df.iloc[-1]["Balance"]
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            summary_results.append({
                "Symbol": symbol, "Timeframe": timeframe_name, "Strategy": strategy_name,
                "Wins": wins, "Losses": losses, "Neither": neither,
                "Final Balance": final_balance, "Win Rate": win_rate,
            })

    summary_df = pd.DataFrame(summary_results)
    summary_filename = "live_trading_summary.csv" if live_trading else "backtest_summary.csv"
    summary_df.to_csv(os.path.join(BACKTEST_SUMMARY_DIR, summary_filename), index=False)
    print(summary_df)


def main(live_trading=False, stop_event=None, config=None):
    os.makedirs(HISTORY_DATA_DIR, exist_ok=True)
    os.makedirs(BACKTEST_SUMMARY_DIR, exist_ok=True)

    if not initialize_mt5():
        return

    if config is None:
        config = {}

    symbols_list = config.get("symbols", trading_pairs.symbols)
    if isinstance(symbols_list, str):
        symbols_list = [s.strip() for s in symbols_list.split(",") if s.strip()]

    try:
        if live_trading:
            while stop_event is None or not stop_event.is_set():
                for symbol in symbols_list:
                    if stop_event is not None and stop_event.is_set():
                        break
                    analyze_symbol(symbol, live_trading=True, config=config)
                # Sleep in 1-second ticks to respond to stop_event immediately
                for _ in range(60):
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(1)
        else:
            for symbol in symbols_list:
                analyze_symbol(symbol, live_trading=False, config=config)
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main(live_trading=False)
