import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
from datetime import datetime, timedelta
import os
import time
import json
import threading

from currency.strategy import Strategy
import trading_pairs
from currency.settings import load_settings, HISTORY_DATA_DIR, BACKTEST_SUMMARY_DIR, ORDER_DB
from currency.lot_size import get_lot_size

# Load global settings
settings = load_settings()

# Constants
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
}
start_time = pd.to_datetime("2024-01-01 00:00:00")
end_time = datetime.now()

def initialize_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code = {mt5.last_error()}")
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

def get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time):
    if not initialize_mt5():
        return
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    if rates is None:
        print(f"No data retrieved for {symbol}, error code = {mt5.last_error()}")
        return
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
    df.to_csv(filename, index=False)

def prep_data(symbol, timeframe_name, visualize=False):
    filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    ohlcv_df = pd.DataFrame(df, columns=["Open", "High", "Low", "Close", "Volume"])
    if visualize:
        mpf.plot(ohlcv_df, type="candle", style="line", title=f"{symbol} {timeframe_name}", volume=True)
    return ohlcv_df

def clean_data(df, symbol, visualize=False):
    symbol_settings = settings.get(symbol, {"polyorder": 2, "window_length": 5})
    polyorder = symbol_settings["polyorder"]
    window_length = symbol_settings["window_length"]
    close_prices = df["Close"].to_numpy()
    smoothed_close = savgol_filter(close_prices, window_length, polyorder)
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
            mpf.make_addplot(df["Is_Low"], scatter=True, markersize=30, marker="v", color="r")
        ]
        mpf.plot(df, type="candle", addplot=apd, style="charles", title=f"{symbol} 1 Hour")

def run_strategy(df, plot_df, RR, initial_balance, risk_amount, risk_type, symbol, live_trading=False):
    balance = mt5.account_info().balance if live_trading else initial_balance
    results, balance_history = [], []
    sent_limits = pd.read_csv(ORDER_DB) if live_trading and os.path.exists(ORDER_DB) else pd.DataFrame(columns=["symbol", "entry_price"])

    wins, losses, neither = 0, 0, 0

    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence

        if live_trading and not sent_limits[(sent_limits['symbol'] == symbol) & (sent_limits['entry_price'] == entry_price)].empty:
            continue

        occurrence_index = df.index.get_loc(occurrence_time)
        trade_df = df.iloc[occurrence_index + 1:]

        entry_reached_mask = trade_df["High"] >= entry_price
        entry_reached_index = trade_df[entry_reached_mask].index.min()

        result = "Pending"
        if pd.notna(entry_reached_index):
            trade_df = trade_df.loc[entry_reached_index:]
            stop_loss_reached = trade_df["High"].ge(stop_loss).idxmax()
            take_profit_reached = trade_df["Low"].le(take_profit).idxmax()

            if stop_loss_reached and (not take_profit_reached or stop_loss_reached < take_profit_reached):
                balance -= risk_amount
                result = "SL"
                losses += 1
            elif take_profit_reached:
                balance += risk_amount * RR
                result = "TP"
                wins += 1
        else:
            neither += 1

        if result == "Pending" and live_trading:
            try:
                volume = get_lot_size(
                    risk_amount=risk_amount,
                    stop_loss=stop_loss,
                    account_currency='USD',
                    symbol=symbol,
                    risk_type=risk_type,
                    account_balance=balance,
                    entry_price=entry_price
                )

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
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    sent_limits = sent_limits._append({"symbol": symbol, "entry_price": entry_price}, ignore_index=True)
                    sent_limits.to_csv(ORDER_DB, index=False)
                    print(sent_limits)
                else:
                    print(f"Failed to place order: {res.retcode}")
            except Exception as e:
                print(f"Error placing order for {symbol}: {e}")


        trade_result = {
            "Occurrence": occurrence_time, "Entry": entry_price, "Stop_Loss": stop_loss,
            "Take_Profit": take_profit, "Result": result, "Balance": balance
        }
        results.append(trade_result)
        balance_history.append({"Occurrence": occurrence_time, "Balance": balance})

    return pd.DataFrame(results), wins, losses, neither

def analyze_symbol(symbol, live_trading=False):
    summary_results = []
    strategies = ["Noir"]

    for timeframe_name, timeframe in timeframes.items():
        get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)
        df = prep_data(symbol, timeframe_name)
        clean_data(df, symbol)
        detect_pivot_points(df, symbol)

        for strategy_name in strategies:
            strategy = Strategy(df)
            plot_df = getattr(strategy, strategy_name)(RR=5)

            initial_balance = 1000
            risk_amount = 25
            risk_type = "fixed"

            backtest_results_df, wins, losses, neither = run_strategy(
                df, plot_df, RR=5, initial_balance=initial_balance,
                risk_amount=risk_amount, risk_type=risk_type, symbol=symbol, live_trading=live_trading
            )

            if not backtest_results_df.empty:
                final_balance = backtest_results_df.iloc[-1]["Balance"]
                win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
                summary_results.append({
                    "Symbol": symbol, "Timeframe": timeframe_name, "Strategy": strategy_name,
                    "Wins": wins, "Losses": losses, "Neither": neither,
                    "Final Balance": final_balance, "Win Rate": win_rate
                })

    summary_df = pd.DataFrame(summary_results)
    summary_filename = "live_trading_summary.csv" if live_trading else "backtest_summary.csv"
    summary_df.to_csv(os.path.join(BACKTEST_SUMMARY_DIR, summary_filename), index=False)
    print(summary_df)

def main(live_trading=False, stop_event=None):
    os.makedirs(HISTORY_DATA_DIR, exist_ok=True)
    os.makedirs(BACKTEST_SUMMARY_DIR, exist_ok=True)

    if not initialize_mt5():
        return

    if live_trading:
        while not stop_event.is_set():
            for symbol in trading_pairs.symbols:
                analyze_symbol(symbol, live_trading=True)
            time.sleep(60)
    else:
        for symbol in trading_pairs.symbols:
            analyze_symbol(symbol, live_trading=False)

if __name__ == "__main__":
    # To run in backtest mode:
    main(live_trading=False)

    # To run in live trading mode:
    # main(live_trading=True)
