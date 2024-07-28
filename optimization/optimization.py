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


def clean_data(df, symbol, visualize=False):
    print(f"Cleaning Data...")
    # Check if symbol exists in settings, otherwise use default settings
    symbol_settings = settings.get(symbol, {"polyorder": 2, "window_length": 5})  # Default values
    
    # Extract settings for the symbol
    polyorder = symbol_settings["polyorder"]
    window_length = symbol_settings["window_length"]
    
    # Ensure df["Close"] is treated as a numpy array for efficiency
    close_prices = df["Close"].to_numpy()
    
    # Apply the Savitzky-Golay filter
    smoothed_close = savgol_filter(close_prices, window_length, polyorder)
    
    # Add smoothed close prices to DataFrame
    df["smoothed_close"] = smoothed_close
    
    print(f'window length = {window_length}')
    print(f'Poly order = {polyorder}')
    
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, close_prices, label="Close Price")
        plt.plot(df.index, smoothed_close, label="Smoothed Close Price")
        plt.legend()
        plt.show()


def detect_pivot_points(df, symbol, visualize=False):
    print("Detecting Pivots...")
    order = settings.get(symbol, {"order": 5})["order"]
    print(f'Order: {order}')
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
    print(f"Running Backtest...😁")

    live_trading = False
    print(f'Live trading: {live_trading}')
    if live_trading:
        if not mt5.initialize():
            print("initialize() failed")
            return

    balance = initial_balance
    balance_history = []
    results = []
    wins = 0
    losses = 0
    neither = 0

    if risk_type == "percentage":
        risk = risk_amount / 100
    else:
        risk = risk_amount

    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence

        occurrence_index = df.index.get_loc(occurrence_time)
        trade_df = df.iloc[occurrence_index + 1:]

        entry_reached_mask = trade_df["High"] >= entry_price
        entry_reached_index = trade_df[entry_reached_mask].index

        if not entry_reached_index.empty:
            entry_reached_index = entry_reached_index[0]
            trade_df = trade_df.loc[entry_reached_index:]

            stop_loss_reached_mask = trade_df["High"] >= stop_loss
            take_profit_reached_mask = trade_df["Low"] <= take_profit

            stop_loss_index = trade_df[stop_loss_reached_mask].index
            take_profit_index = trade_df[take_profit_reached_mask].index

            if not stop_loss_index.empty and not take_profit_index.empty:
                if stop_loss_index[0] < take_profit_index[0]:
                    balance -= risk * balance if risk_type == "percentage" else risk
                    result = "SL"
                    losses += 1
                else:
                    balance += (risk * RR * balance) if risk_type == "percentage" else (risk * RR)
                    result = "TP"
                    wins += 1
            elif not stop_loss_index.empty:
                balance -= risk * balance if risk_type == "percentage" else risk
                result = "SL"
                losses += 1
            elif not take_profit_index.empty:
                balance += (risk * RR * balance) if risk_type == "percentage" else (risk * RR)
                result = "TP"
                wins += 1
            else:
                result = "Running"
        else:
            result = "Pending"
            neither += 1

            if live_trading:
                volume = 0.5  # Adjust this as needed

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
                    "comment": "EchelNet Bot",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                res = mt5.order_send(request)
                if res.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to place order: {res.retcode}")

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
    final_balance = backtest_df.iloc[-1]["Balance"]
    balance_df = pd.DataFrame(balance_history)
    max_daily_drawdown, worst_day = calculate_max_daily_drawdown(balance_df)
    print(f"Starting Balance: {initial_balance}")
    print(f"Max Drawdown Amount: {max_drawdown_amount}")
    print(f"Max Drawdown Percentage: {max_drawdown_percentage * 100:.2f}%")
    print(f"Max Daily Drawdown: {max_daily_drawdown * 100:.2f}% on {worst_day}")
    print(f'Final Balance: {final_balance}')

    print()
    print()
    print()


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

            for polyorder in range(2, 15):
                for window_length in range(2, 20):
                    for order in range(2, 15):
                        if polyorder < window_length:
                            settings[symbol] = {
                                "polyorder": polyorder,
                                "window_length": window_length,
                                "order": order
                            }

                            def process_data():
                                df = prep_data(symbol, timeframe_name, visualize=False)
                                clean_data(df, symbol)
                                detect_pivot_points(df, symbol)
                                return df

                            df = process_data()

                            print(f"Running backtest for {symbol} on {timeframe_name} timeframe with polyorder {polyorder} and window length {window_length}...")

                            for strategy_name in strategies:
                                strategy = Strategy(df)
                                plot_df = getattr(strategy, strategy_name)(RR=5)

                                initial_balance = 100
                                risk_amount = 10
                                risk_type = "fixed"

                                results = backtest(df, plot_df, RR=5, initial_balance=initial_balance, risk_amount=risk_amount, risk_type=risk_type, symbol=symbol)

                                backtest_results_df, wins, losses, neither, max_drawdown_amount, max_drawdown_percentage, max_daily_drawdown, worst_day = results

                                final_balance = backtest_results_df.iloc[-1]["Balance"]
                                win_rate = (wins / (wins + losses)) * 100 if wins != 0 else 0

                                current_count = highest_count = 0
                                for i in range(1, len(backtest_results_df)):
                                    if backtest_results_df.iloc[i]["Result"] == backtest_results_df.iloc[i - 1]["Result"] == "SL":
                                        current_count += 1
                                        highest_count = max(highest_count, current_count)
                                    else:
                                        current_count = 1

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

                                backtest_results_df["Symbol"] = symbol
                                all_backtest_results.append(backtest_results_df)
                                optimization_df = pd.DataFrame(summary_results)
                                optimization_df.to_csv(os.path.join(backtest_summary_dir, f'Opt_{symbol}_summary.csv'), index=True)

            # plot_balance_graph(backtest_results_df)
            combined_backtest_df = pd.concat(all_backtest_results)
            combined_backtest_df.to_csv(os.path.join(backtest_summary_dir, "Combined_resuts.csv"), index=False)

    summary_df = pd.DataFrame(summary_results)
    print(summary_df)
    summary_df.to_csv(os.path.join(backtest_summary_dir, "Complete_Optimization.csv"), index=False)
    shutdown_mt5()
if __name__ == "__main__":
    main()

