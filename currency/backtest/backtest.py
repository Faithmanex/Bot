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
import trendet
import trading_pairs
import json

def load_settings(settings_file="settings.json"):
    with open(settings_file, "r") as file:
        settings = json.load(file)
    
    # Define default settings for symbols
    default_symbol_settings = {
        "polyorder": 8,
        "window_length": 15,
        "order": 9
    }

    # Ensure every symbol has settings, using defaults if necessary
    for symbol in trading_pairs.symbols:
        if symbol not in settings:
            settings[symbol] = default_symbol_settings
    
    return settings


settings = load_settings()

# Constants
history_data_dir = "history_data"
backtest_summary_dir = "backtest_summary"
symbols = trading_pairs.symbols
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
}
start_time = pd.to_datetime("2024-07-01 00:00:00")
end_time = pd.to_datetime("2024-07-23 23:00:00")

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
    # Check if symbol exists in settings, otherwise use default settings
    symbol_settings = settings.get(symbol)
    
    polyorder = symbol_settings["polyorder"]
    window_length = symbol_settings["window_length"]
    
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
    print(f'Order: {order}')
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

    live_trading = True
    print(f'Live trading: {live_trading}')
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
                result = "Running"
        else:
            result = "Pending"

        if result == "Pending":
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
    # print(backtest_df)

    balance_series = pd.Series([entry['Balance'] for entry in balance_history])
    max_drawdown_amount, max_drawdown_percentage = calculate_drawdown(balance_series)

    balance_df = pd.DataFrame(balance_history)
    max_daily_drawdown, worst_day = calculate_max_daily_drawdown(balance_df)
    print(f"Starting Balance: {initial_balance}")
    print(f"Max Drawdown Amount: {max_drawdown_amount}")
    print(f"Max Drawdown Percentage: {max_drawdown_percentage * 100:.2f}%")
    print(f"Max Daily Drawdown: {max_daily_drawdown * 100:.2f}% on {worst_day}")

    mt5.shutdown()

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
    # Initialization and setup (unchanged)
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return
    
    symbols = mt5.symbols_get(group="ForexMinor")
    if symbols is not None:
        print("Available Forex Minor Pairs:")
        for symbol in symbols:
            print(symbol.name)

    summary_results = []
    strategies = ["Noir"]
    all_backtest_results = []  # Added to store backtest results of all symbols

    for symbol in trading_pairs.symbols:
        for timeframe_name, timeframe in timeframes.items():
            print()
            print()
            print(f"Running backtest for {symbol} on {timeframe_name} timeframe...")

            # Data preparation and cleaning (unchanged)
            def process_data_wrapper():
                global df
                get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)
                df = prep_data(symbol, timeframe_name, visualize=False)
                clean_data(df, symbol)
                detect_pivot_points(df, symbol)

            thread_process_data = threading.Thread(target=process_data_wrapper)
            thread_process_data.start()
            thread_process_data.join()

            for strategy_name in strategies:
                strategy = Strategy(df)
                plot_df = getattr(strategy, strategy_name)(RR=5)

                initial_balance = 10000
                risk_amount = 50
                risk_type = "fixed"
                backtest_results_df, wins, losses, neither, max_drawdown_amount, max_drawdown_percentage, max_daily_drawdown, worst_day = backtest(
                    df, plot_df, RR=5, initial_balance=initial_balance, risk_amount=risk_amount, risk_type=risk_type, symbol=symbol
                )

                final_balance = backtest_results_df.iloc[-1]["Balance"]
                if wins != 0:
                    win_rate = (wins / (wins + losses)) * 100
                else:
                    win_rate = 0

                current_count = 0
                highest_count = 0

                for i in range(len(backtest_results_df)):
                    if (
                        backtest_results_df.iloc[i]["Result"]
                        == backtest_results_df.iloc[i - 1]["Result"]
                        == "SL"
                    ):
                        current_count += 1
                        if current_count > highest_count:
                            highest_count = current_count
                    else:
                        current_count = 1

                summary_results.append({
                    "Symbol": symbol,
                    "Timeframe": timeframe_name,
                    "Strategy": strategy_name,
                    "Wins": wins,
                    "Losses": losses,
                    "Neither": neither,
                    "Consecutive SL": highest_count,
                    "Final Balance": final_balance,
                    "Win Rate": win_rate,
                    "Max Drawdown Amount": max_drawdown_amount,
                    "Max Drawdown Percentage": max_drawdown_percentage * 100,
                    "Max Daily Drawdown": max_daily_drawdown * 100,
                    "Worst Day": worst_day
                })
                print(f"Final Balance: {final_balance}")
                # Append the backtest results of the current symbol to the all_backtest_results list
                backtest_results_df["Symbol"] = symbol  # Add a column to identify the symbol
                all_backtest_results.append(backtest_results_df)  # Added to collect results
        # plot_balance_graph(backtest_results_df)
    # Combine all backtest results into a single DataFrame and save to CSV
    combined_backtest_df = pd.concat(all_backtest_results)  # Combine all results
    combined_backtest_df.to_csv(os.path.join(backtest_summary_dir, "all_backtest_results.csv"), index=False)  # Save to CSV

    summary_df = pd.DataFrame(summary_results)
    print(summary_df)
    summary_df.to_csv(os.path.join(backtest_summary_dir, "backtest_summary.csv"), index=False)

    # Uncomment to plot individual trades
    # for trade in plot_df.itertuples():
    #     plot(trade, df, symbol)

if __name__ == "__main__":
    main()
