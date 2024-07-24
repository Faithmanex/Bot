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
from concurrent.futures import ThreadPoolExecutor

from strategy import Strategy  # Assuming this is your custom strategy module
import trendet
import trading_pairs

# Constants
history_data_dir = "history_data"
backtest_summary_dir = "backtest_summary"
order_db = "sent_limits.csv"
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
}
start_time = pd.to_datetime("2024-07-01 00:00:00")
end_time = datetime.now()

# Ensure directories exist
os.makedirs(history_data_dir, exist_ok=True)
os.makedirs(backtest_summary_dir, exist_ok=True)

# Default settings
default_settings = {
    "window_length": 15,
    "polyorder": 8,
    "order": 3
}

def get_symbol_settings(symbol):
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
            return settings.get(symbol, default_settings)
    except FileNotFoundError:
        return default_settings
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for symbol {symbol}: {e}")
        return default_settings

def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

def shutdown_mt5():
    mt5.shutdown()

def get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time):
    try:
        end_time = datetime.now()
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

        if rates is None:
            print(f"No data retrieved for {symbol}, error code =", mt5.last_error())
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
        df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error getting historical data for {symbol}: {e}")

def update_historical_data(symbol, timeframe, timeframe_name):
    try:
        initialize_mt5()
        latest_time = pd.to_datetime(mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)[0][0], unit='s')
        new_end_time = latest_time - timedelta(minutes=1)
        rates = mt5.copy_rates_range(symbol, timeframe, latest_time, new_end_time)

        if rates is not None:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
            if os.path.exists(filename):
                df.to_csv(filename, mode='a', header=False, index=False)
            else:
                df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error updating historical data for {symbol}: {e}")

def prep_data(symbol, timeframe_name, visualize=False):
    try:
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
    except FileNotFoundError as e:
        print(f"File not found for {symbol}: {e}")
    except Exception as e:
        print(f"Error preparing data for {symbol}: {e}")

def clean_data(df, symbol, visualize=False):
    try:
        settings = get_symbol_settings(symbol)
        window_length = settings["window_length"]
        polyorder = settings["polyorder"]

        smoothed_close = savgol_filter(df["Close"], window_length, polyorder)
        df["smoothed_close"] = smoothed_close

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df["Close"], label="Close Price")
            plt.plot(df.index, df["smoothed_close"], label="Smoothed Close Price")
            plt.legend()
            plt.show()
    except Exception as e:
        print(f"Error cleaning data for {symbol}: {e}")

def detect_pivot_points(df, symbol, visualize=False):
    try:
        settings = get_symbol_settings(symbol)
        order = settings["order"]

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
    except Exception as e:
        print(f"Error detecting pivot points for {symbol}: {e}")

def initialize_sent_limits():
    try:
        if os.path.exists(order_db):
            # Clear the file content
            with open(order_db, 'w') as file:
                file.write('symbol,entry_price\n')  # Write headers to the file
        else:
            # Create a new file with headers
            pd.DataFrame(columns=["symbol", "entry_price"]).to_csv(order_db, index=False)
    except Exception as e:
        print(f"Error initializing sent limits: {e}")

def backtest(df, plot_df, RR, initial_balance, risk_amount, risk_type, symbol):
    live_trading = True

    try:
        if live_trading:
            initialize_mt5()
            # Assuming mt5.get_account_info() returns an object with a 'balance' attribute
            balance = mt5.account_info().balance
        else:
            balance = initial_balance
    except Exception as e:
        print(f"An error occurred while getting account balance: {e}")
        return pd.DataFrame(), 0, 0, 0

    results = []
    wins = 0
    losses = 0
    neither = 0
    balance_history = []

    sent_limits = pd.read_csv(order_db) if os.path.exists(order_db) else pd.DataFrame(columns=["symbol", "entry_price"])
    
    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence
        # print(sent_limits)
        if live_trading and not sent_limits[(sent_limits['symbol'] == symbol) & (sent_limits['entry_price'].astype(float) == entry_price)].empty:
            continue

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

        if result == "Pending" and live_trading:
            from lot_size import get_lot_size  # Import the get_lot_size function from lot_size module

            try:
                volume = get_lot_size(
                    risk_amount=risk_amount,
                    stop_loss=stop_loss,
                    account_currency='USD',
                    symbol=symbol,
                    risk_type=risk_type,
                    account_balance=balance,
                    entry_price = entry_price
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
                    "comment": "Ebot",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    sent_limits = sent_limits._append({"symbol": symbol, "entry_price": entry_price}, ignore_index=True)
                    print(sent_limits)
                    sent_limits.to_csv(order_db, index=False)
                    print(sent_limits)
                else:
                    print(f"Failed to place order: {res.retcode}")
            except Exception as e:
                print(f"Error placing order for {symbol}: {e}")

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

    return pd.DataFrame(results), wins, losses, neither

def analyze_symbol(symbol):
    summary_results = []
    strategies = ["AMSstrategy"]

    for timeframe_name, timeframe in timeframes.items():
        print(f"Running live trading for {symbol} on {timeframe_name} timeframe...")

        try:
            get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)

            while True:
                try:
                    get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)
                    df = prep_data(symbol, timeframe_name, visualize=False)
                    clean_data(df, symbol)
                    detect_pivot_points(df, symbol)
                    # print(df)
                    
                    with ThreadPoolExecutor() as executor:
                        for strategy_name in strategies:
                            strategy = Strategy(df)
                            plot_df = getattr(strategy, strategy_name)(RR=5)

                            initial_balance = mt5.account_info().balance
                            risk_amount = 25
                            risk_type = "fixed"
                            backtest_results_df, wins, losses, neither = backtest(df, plot_df, RR=5, initial_balance=initial_balance, risk_amount=risk_amount, risk_type=risk_type, symbol=symbol)

                            final_balance = backtest_results_df.iloc[-1]["Balance"]
                            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) != 0 else 0

                            current_count = 0
                            highest_count = 0

                            for i in range(1, len(backtest_results_df)):  # Fixed index start for checking previous result
                                if backtest_results_df.iloc[i]["Result"] == backtest_results_df.iloc[i - 1]["Result"] == "SL":
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
                            })

                    summary_df = pd.DataFrame(summary_results)
                    print(summary_df)
                    summary_df.to_csv(os.path.join(backtest_summary_dir, "live_trading_summary.csv"), index=False)

                except Exception as e:
                    print(f"An error occurred while analyzing {symbol}: {e}")

                time.sleep(180)  # Wait for 5 minutes before the next update

        except Exception as e:
            print(f"An error occurred while setting up analysis for {symbol}: {e}")

def main():
    try:
        initialize_mt5()
        initialize_sent_limits()  # Initialize the sent limits file
        symbols = trading_pairs.symbols

        with ThreadPoolExecutor() as executor:
            executor.map(analyze_symbol, symbols)
    except Exception as e:
        print(f"An error occurred in the main function: {e}")
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()