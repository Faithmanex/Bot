from strategy import Strategy
import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
from datetime import datetime, timedelta
import os
import pandas_ta as ta
import threading
import trading_pairs
import json

def load_settings(settings_file="settings.json"):
    with open(settings_file, "r") as file:
        settings = json.load(file)
    
    # Define default settings for symbols
    default_symbol_settings = {}
    for polyorder in range(2, 16):
        for window_length in range(2, 16):
            if polyorder < window_length:
                default_symbol_settings[f"{polyorder}_{window_length}"] = {
                    "polyorder": polyorder,
                    "window_length": window_length,
                    "order": 9
                }

    # Ensure every symbol has settings, using defaults if necessary
    for symbol in trading_pairs.symbols:
        if symbol not in settings:
            settings[symbol] = default_symbol_settings[f"2_15"]  # Using 2_15 as the default

    return settings


settings = load_settings()

# Constants
history_data_dir = "history_data"
backtest_summary_dir = "backtest_summary"
symbols = trading_pairs.symbols
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
}

start_time = pd.to_datetime("2024-01-01 00:00:00")
end_time = datetime.now()

# Ensure directories exist
os.makedirs(history_data_dir, exist_ok=True)
os.makedirs(backtest_summary_dir, exist_ok=True)

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
        print(f"No data retrieved, error code = {mt5.last_error()}")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
    df.to_csv(filename, index=False)



def prep_data(symbol, timeframe_name, visualize=False):
    print(f"Preparing Data...")
    filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    ohlcv_data = df[["open", "high", "low", "close", "tick_volume"]].to_numpy()
    ohlcv_df = pd.DataFrame(ohlcv_data, columns=["Open", "High", "Low", "Close", "Volume"], index=df.index)
    
    if visualize:
        mpf.plot(ohlcv_df, type="candle", style="line", title=f"{symbol} {timeframe_name}", volume=True)
    
    return ohlcv_df


def clean_data(df, polyorder, window_length):
    close_prices = df["Close"].to_numpy()
    smoothed_close = savgol_filter(close_prices, window_length, polyorder)
    df["smoothed_close"] = smoothed_close
    return df


def detect_pivot_points(df, order):
    smoothed_close = df["smoothed_close"].to_numpy()
    highs = argrelextrema(smoothed_close, np.greater, mode="wrap", order=order)[0]
    lows = argrelextrema(smoothed_close, np.less, mode="wrap", order=order)[0]

    df["Is_High"] = np.nan
    df["Is_Low"] = np.nan
    df.loc[df.index[highs], "Is_High"] = df["High"].iloc[highs]
    df.loc[df.index[lows], "Is_Low"] = df["Low"].iloc[lows]
    return df


def calculate_drawdown(balance_series):
    balance_array = balance_series.to_numpy()
    drawdown = (np.maximum.accumulate(balance_array) - balance_array) / np.maximum.accumulate(balance_array)
    max_drawdown = drawdown.max()
    max_drawdown_amount = (np.maximum.accumulate(balance_array) - balance_array).max()
    return max_drawdown_amount, max_drawdown


def calculate_max_daily_drawdown(balance_df):
    balance_df['Date'] = balance_df['Occurrence'].dt.date
    daily_max_balance = balance_df.groupby('Date')['Balance'].max()
    daily_min_balance = balance_df.groupby('Date')['Balance'].min()
    daily_drawdown = (daily_max_balance - daily_min_balance) / daily_max_balance
    max_daily_drawdown = daily_drawdown.max()
    return max_daily_drawdown, daily_drawdown.idxmax()

def backtest(df, plot_df, RR, initial_balance, risk_amount, risk_type, symbol):
    balance = initial_balance
    results = []
    balance_history = []
    wins = 0
    losses = 0
    neither = 0

    if plot_df.empty:
        return pd.DataFrame(), 0, 0, 0, 0.0, 0.0, 0.0, None

    entries = plot_df['Entry'].values
    stop_losses = plot_df['Stop_Loss'].values
    take_profits = plot_df['Take_Profit'].values
    occurrences = plot_df['Occurence'].values

    high_prices = df['High'].values
    low_prices = df['Low'].values

    for idx, entry_price in enumerate(entries):
        stop_loss = stop_losses[idx]
        take_profit = take_profits[idx]
        occurrence_time = occurrences[idx]

        occurrence_index = df.index.get_loc(occurrence_time)
        entry_reached = False

        if risk_type == "percentage":
            Risk = (risk_amount / 100) * balance
        else:
            Risk = risk_amount

        subsequent_highs = high_prices[occurrence_index + 1:]
        subsequent_lows = low_prices[occurrence_index + 1:]

        entry_reached_mask = subsequent_highs >= entry_price
        if np.any(entry_reached_mask):
            entry_reached = True
            first_entry_index = np.argmax(entry_reached_mask)

            stop_loss_reached_mask = subsequent_highs[first_entry_index:] >= stop_loss
            take_profit_reached_mask = subsequent_lows[first_entry_index:] <= take_profit

            if np.any(stop_loss_reached_mask):
                stop_loss_reached_index = np.argmax(stop_loss_reached_mask)
            else:
                stop_loss_reached_index = len(subsequent_highs)

            if np.any(take_profit_reached_mask):
                take_profit_reached_index = np.argmax(take_profit_reached_mask)
            else:
                take_profit_reached_index = len(subsequent_lows)

            if stop_loss_reached_index < take_profit_reached_index:
                balance -= Risk
                result = "SL"
                losses += 1
            elif take_profit_reached_index < stop_loss_reached_index:
                balance += Risk * RR
                result = "TP"
                wins += 1
            else:
                result = "Running"
        else:
            result = "Pending"
            neither += 1

        trade_result = {
            "Occurrence": occurrence_time,
            "Entry": entry_price,
            "Stop_Loss": stop_loss,
            "Take_Profit": take_profit,
            "Result": result,
            "Balance": balance,
        }

        balance_history.append({"Occurrence": occurrence_time, "Balance": balance})
        results.append(trade_result)

    backtest_df = pd.DataFrame(results)

    balance_series = pd.Series([entry['Balance'] for entry in balance_history])
    max_drawdown_amount, max_drawdown_percentage = calculate_drawdown(balance_series)

    balance_df = pd.DataFrame(balance_history)
    max_daily_drawdown, worst_day = calculate_max_daily_drawdown(balance_df)

    return backtest_df, wins, losses, neither, max_drawdown_amount, max_drawdown_percentage, max_daily_drawdown, worst_day

def plot_balance_graph(backtest_results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(backtest_results_df["Occurrence"], backtest_results_df["Balance"], marker="")
    plt.title("ACCOUNT GROWTH")
    plt.xlabel("Trade Time")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.show()

def plot(trade, df, symbol):
    specific_datetime = trade.Occurence
    dfpl = df[specific_datetime - timedelta(days=7):specific_datetime + timedelta(days=20)]
    # apd = [
    #     mpf.make_addplot(dfpl["Is_High"], scatter=True, markersize=30, marker="x", color="b"),
    #     mpf.make_addplot(dfpl["Is_Low"], scatter=True, markersize=30, marker="x", color="r"),
    # ]
    mpf.plot(
        dfpl,
        type="candle",
        style="nightclouds",
        title=f"{symbol}",
        warn_too_much_data=9999999999,
        # addplot=apd,
        vlines=[specific_datetime],
        hlines=dict(hlines=[trade.Stop_Loss, trade.Take_Profit, trade.Entry], colors=['r','g','b'], linestyle='-'),
    )

def main():
    if not initialize_mt5():
        return
    
    symbols = mt5.symbols_get(group="ForexMinor")
    if symbols:
        print("Available Forex Minor Pairs:")
        for symbol in symbols:
            print(symbol.name)

    all_backtest_results = []
    summary_results = []
    strategies = ["Noir"]

    for symbol in trading_pairs.symbols:
        for timeframe_name, timeframe in timeframes.items():
            print(f"\nFetching historical data for {symbol} on {timeframe_name} timeframe...")
            get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)

            # Load the raw dataframe once per symbol/timeframe
            df_raw = prep_data(symbol, timeframe_name, visualize=False)

            for polyorder in range(2, 15):
                for window_length in range(2, 20):
                    if polyorder >= window_length:
                        continue

                    # Apply smoothing once per (polyorder, window_length)
                    df_smoothed = df_raw.copy()
                    df_smoothed = clean_data(df_smoothed, polyorder, window_length)

                    for order in range(2, 15):
                        # Detect pivot points for the current order
                        df_pivots = df_smoothed.copy()
                        df_pivots = detect_pivot_points(df_pivots, order)

                        print(f"Running backtest for {symbol} on {timeframe_name} timeframe with polyorder {polyorder}, window length {window_length}, order {order}...")

                        for strategy_name in strategies:
                            strategy = Strategy(df_pivots)
                            plot_df = getattr(strategy, strategy_name)(RR=5)

                            initial_balance = 100
                            risk_amount = 10
                            risk_type = "fixed"

                            results = backtest(df_pivots, plot_df, RR=5, initial_balance=initial_balance, risk_amount=risk_amount, risk_type=risk_type, symbol=symbol)
                            backtest_results_df, wins, losses, neither, max_drawdown_amount, max_drawdown_percentage, max_daily_drawdown, worst_day = results

                            if backtest_results_df.empty:
                                final_balance = initial_balance
                                highest_count = 0
                                win_rate = 0.0
                            else:
                                final_balance = backtest_results_df.iloc[-1]["Balance"]
                                win_rate = (wins / (wins + losses)) * 100 if wins != 0 else 0

                                current_count = 0
                                highest_count = 0
                                for i in range(len(backtest_results_df)):
                                    if backtest_results_df.iloc[i]["Result"] == "SL":
                                        current_count += 1
                                        highest_count = max(highest_count, current_count)
                                    else:
                                        current_count = 0

                            summary_results.append({
                                "Symbol": symbol,
                                "Wins": wins,
                                "Losses": losses,
                                "Neither": neither,
                                "Consecutive SL": highest_count,
                                "Final Balance": final_balance,
                                "Win Rate": win_rate,
                                "P": polyorder,
                                "W": window_length,
                                "O": order
                            })

                            if not backtest_results_df.empty:
                                backtest_results_df["Symbol"] = symbol
                                all_backtest_results.append(backtest_results_df)

            # Save the individual symbol's optimization summary once at the end of its run
            optimization_df = pd.DataFrame(summary_results)
            optimization_df.to_csv(os.path.join(backtest_summary_dir, f'Opt_{symbol}_summary.csv'), index=True)

            # plot_balance_graph(backtest_results_df)
            if all_backtest_results:
                combined_backtest_df = pd.concat(all_backtest_results)
                combined_backtest_df.to_csv(os.path.join(backtest_summary_dir, "Combined_resuts.csv"), index=False)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        print(summary_df)
        summary_df.to_csv(os.path.join(backtest_summary_dir, "Complete_Optimization.csv"), index=False)
    shutdown_mt5()
if __name__ == "__main__":
    main()

