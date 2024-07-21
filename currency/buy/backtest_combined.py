import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
from datetime import datetime, timedelta
import os
import pandas_ta as ta
import threading
import trendet
import trading_pairs
import json

def load_settings(settings_file="settings.json"):
    with open(settings_file, "r") as file:
        settings = json.load(file)
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
end_time = pd.to_datetime("2024-07-14 23:00:00")

# Ensure directories exist
os.makedirs(history_data_dir, exist_ok=True)
os.makedirs(backtest_summary_dir, exist_ok=True)

def get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

    if rates is None:
        print("No data retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        quit()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
    df.to_csv(filename, index=False)
    mt5.shutdown()

def prep_data(symbol, timeframe_name, visualize=False):
    print(f"Preparing Data...")
    filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    ohlcv_data = df[["open", "high", "low", "close", "tick_volume"]]
    ohlcv_data.columns = ["Open", "High", "Low", "Close", "Volume"]
    if visualize:
        mpf.plot(ohlcv_data, type="candle", style="line", title=f"{symbol} {timeframe_name}", volume=True)
        return ohlcv_data
    else:
        return ohlcv_data

def clean_data(df, symbol, visualize=False):
    print(f"Cleaning Data...")
    polyorder = settings[symbol]["polyorder"]
    window_length = settings[symbol]["window_length"]
    smoothed_close = savgol_filter(df["Close"], window_length, polyorder)
    df["smoothed_close"] = smoothed_close
    print(f'window length = {window_length}')
    print(f'Poly order = {polyorder}')
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["Close"], label="Close Price")
        plt.plot(df.index, df["smoothed_close"], label="Smoothed Close Price")
        plt.legend()
        plt.show()

def detect_pivot_points(df, symbol, visualize=False):
    print(f"Detecting Pivots...")
    order = settings[symbol]["order"]
    print(order)
    highs = argrelextrema(df["smoothed_close"].to_numpy(), np.greater, mode="wrap", order=order)
    lows = argrelextrema(df["smoothed_close"].to_numpy(), np.less, mode="wrap", order=order)

    df["Is_High"] = df["High"].iloc[highs[0]]
    df["Is_Low"] = df["Low"].iloc[lows[0]]
    df.fillna(0)

    if visualize:
        apd = [
            mpf.make_addplot(df["Is_High"], scatter=True, markersize=30, marker="^", color="g"),
            mpf.make_addplot(df["Is_Low"], scatter=True, markersize=30, marker="v", color="r")
        ]
        mpf.plot(df, type="candle", addplot=apd, style="charles", title=f"{symbol} 1 Hour")


def calculate_supertrend(df, period=5, multiplier=3):
    """
    Function to calculate the SuperTrend indicator
    """
    df['ATR'] = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=period)
    df['Upper Band'] = ((df['High'] + df['Low']) / 2) + (multiplier * df['ATR'])
    df['Lower Band'] = ((df['High'] + df['Low']) / 2) - (multiplier * df['ATR'])
    df['In Uptrend'] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df['Close'][current] > df['Upper Band'][previous]:
            df['In Uptrend'][current] = True
        elif df['Close'][current] < df['Lower Band'][previous]:
            df['In Uptrend'][current] = False
        else:
            df['In Uptrend'][current] = df['In Uptrend'][previous]

            if df['In Uptrend'][current] and df['Lower Band'][current] < df['Lower Band'][previous]:
                df['Lower Band'][current] = df['Lower Band'][previous]

            if not df['In Uptrend'][current] and df['Upper Band'][current] > df['Upper Band'][previous]:
                df['Upper Band'][current] = df['Upper Band'][previous]

    df['SuperTrend'] = df.apply(lambda row: 1 if row['In Uptrend'] else 0, axis=1)
    return df

def identify_trends(df, start_time, end_time):
    trends_df = trendet.identify_df_trends(
        df,
        column='Close',
        identify='both'
    )
    
    trend_map = {'Up Trend': 1, 'Down Trend': -1, 'No Trend': 0}
    trends_df['Trend'] = trends_df['Up Trend'].map(trend_map).fillna(trends_df['Down Trend'].map(trend_map))
    df['Trend'] = trends_df['Trend']
    trends_df.to_csv(os.path.join(backtest_summary_dir, "trend.csv"), index=False)
    print(trends_df)
    return df

def calculate_drawdown(balance_series):
    drawdown = (balance_series.cummax() - balance_series) / balance_series.cummax()
    max_drawdown = drawdown.max()
    max_drawdown_amount = (balance_series.cummax() - balance_series).max()
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
    print(live_trading)
    if live_trading:
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
            return

    results = []
    wins = 0
    losses = 0
    neither = 0
    balance = initial_balance
    balance_history = []

    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence

        price_reached_stop_loss = False
        price_reached_take_profit = False

        occurrence_index = df.index.get_loc(occurrence_time)

        entry_reached = False
        if risk_type == "percentage":
            Risk = risk_amount / 100 * balance
        else:
            Risk = risk_amount

        for i in range(occurrence_index + 1, len(df)):
            high_price = df.iloc[i]["High"]
            low_price = df.iloc[i]["Low"]

            if not entry_reached and high_price >= entry_price:
                entry_reached = True

            if entry_reached:
                if high_price >= stop_loss:
                    price_reached_stop_loss = True
                    balance -= Risk
                    result = "SL"
                    losses += 1
                    break

                if low_price <= take_profit:
                    price_reached_take_profit = True
                    balance += Risk * RR
                    result = "TP"
                    wins += 1
                    break

        if entry_reached:
            if not (price_reached_stop_loss or price_reached_take_profit):
                result = "N"
                neither += 1
            balance_history.append(
                {"Occurrence": occurrence_time, "Balance": balance}
            )
            results.append(
                {
                    "Entry": entry_price,
                    "Stop Loss": stop_loss,
                    "Take Profit": take_profit,
                    "Result": result,
                    "Balance": balance,
                    "Occurrence": occurrence_time,
                }
            )
        else:
            neither += 1

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(backtest_summary_dir, f"{symbol}_backtest_results.csv"), index=False)
    result_df["Occurrence"] = pd.to_datetime(result_df["Occurrence"])
    result_df.set_index("Occurrence", inplace=True)

    balance_df = pd.DataFrame(balance_history)
    balance_df["Occurrence"] = pd.to_datetime(balance_df["Occurrence"])

    max_drawdown_amount, max_drawdown = calculate_drawdown(result_df["Balance"])
    max_daily_drawdown, max_daily_drawdown_date = calculate_max_daily_drawdown(balance_df)

    backtest_summary = {
        "Symbol": symbol,
        "Initial Balance": initial_balance,
        "Final Balance": balance,
        "Total Trades": len(result_df),
        "Wins": wins,
        "Losses": losses,
        "Neither": neither,
        "Max Drawdown Amount": max_drawdown_amount,
        "Max Drawdown Percentage": max_drawdown,
        "Max Daily Drawdown": max_daily_drawdown,
        "Max Daily Drawdown Date": max_daily_drawdown_date,
    }

    with open(os.path.join(backtest_summary_dir, f"{symbol}_backtest_summary.json"), "w") as file:
        json.dump(backtest_summary, file, indent=4)

    if not result_df.empty:
        plt.plot(result_df.index, result_df["Balance"], label="Balance")
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.title(f"Balance Over Time for {symbol}")
        plt.legend()
        plt.show()

    print(f"Backtest summary for {symbol}:")
    print(json.dumps(backtest_summary, indent=4))
    print(f"Backtest completed...")

    return balance, result_df

def plot_combined_performance(summaries):
    plt.figure(figsize=(12, 8))
    for symbol, summary in summaries.items():
        balance_series = summary["balance_series"]
        plt.plot(balance_series.index, balance_series, label=symbol)

    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.title("Combined Performance Over Time")
    plt.legend()
    plt.show()

def run_backtest_for_all_symbols(symbols, timeframe, timeframe_name, start_time, end_time, RR, initial_balance, risk_amount, risk_type):
    combined_summary = {}

    for symbol in symbols:
        get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)
        df = prep_data(symbol, timeframe_name)
        clean_data(df, symbol)
        detect_pivot_points(df, symbol)
        supertrend_df = calculate_supertrend(df)
        trends_df = identify_trends(df, start_time, end_time)

        balance, backtest_result_df = backtest(df, supertrend_df, RR, initial_balance, risk_amount, risk_type, symbol)
        combined_summary[symbol] = {
            "final_balance": balance,
            "balance_series": backtest_result_df["Balance"]
        }

    plot_combined_performance(combined_summary)

# Example Usage
run_backtest_for_all_symbols(
    symbols=symbols,
    timeframe=mt5.TIMEFRAME_H1,
    timeframe_name="H1",
    start_time=start_time,
    end_time=end_time,
    RR=2,
    initial_balance=10000,
    risk_amount=100,
    risk_type="percentage"
)
