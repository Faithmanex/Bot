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
from concurrent.futures import ProcessPoolExecutor

from strategy import Strategy  # Custom strategy module imported

import trading_pairs

# Constants for directory paths and timeframes
history_data_dir = "history_data"
backtest_summary_dir = "backtest_summary"
order_db = "sent_limits.csv"
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
}

# Time range for backtesting
start_time = pd.to_datetime("2024-01-01 00:00:00")
end_time = datetime.now()

# Ensure necessary directories exist
os.makedirs(history_data_dir, exist_ok=True)
os.makedirs(backtest_summary_dir, exist_ok=True)

def load_settings(settings_file="c:/Users/DELL XPS 9360/Documents/GitHub/Bot/currency/backtest/settings.json"):
    """
    Load settings from a JSON file. If the file does not exist or cannot be decoded, 
    default settings are applied for each symbol.
    """
    try:
        with open(settings_file, "r") as file:
            settings = json.load(file)
    except FileNotFoundError:
        print(f"Settings file not found: {settings_file}")
        settings = {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        settings = {}

    # Default settings for symbols
    default_symbol_settings = {
        "polyorder": 8,
        "window_length": 15,
        "order": 3
    }

    # Apply default settings if necessary
    for symbol in trading_pairs.symbols:
        if symbol not in settings:
            settings[symbol] = default_symbol_settings
    
    return settings

# Load global settings
settings = load_settings()

def initialize_mt5():
    """
    Initialize MetaTrader5 connection. Exits the script if initialization fails.
    """
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

def shutdown_mt5():
    """
    Shutdown MetaTrader5 connection gracefully.
    """
    mt5.shutdown()

def get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time):
    """
    Retrieve historical data for a given symbol and timeframe, then save it to a CSV file.
    
    Parameters:
    - symbol: Trading symbol to retrieve data for.
    - timeframe: Timeframe for the historical data.
    - timeframe_name: Name of the timeframe (used in the filename).
    - start_time: Start time for the historical data.
    - end_time: End time for the historical data.
    """
    initialize_mt5()  # Ensure MT5 is initialized before calling this function
    
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        if rates is None:
            print(f"No data retrieved for {symbol}, error code = {mt5.last_error()}")
            return
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        filename = os.path.join(history_data_dir, f"{symbol}_data_{timeframe_name}.csv")
        df.to_csv(filename, index=False)
        
    except Exception as e:
        print(f"Error retrieving historical data for {symbol}: {e}")
 # Ensure MT5 is shutdown after retrieving the data


def update_historical_data(symbol, timeframe, timeframe_name):
    """
    Update existing historical market data for a given symbol and timeframe. 
    New data is appended to the CSV file in the history_data directory.
    """
    try:
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
    """
    Clean market data by applying a Savitzky-Golay filter to smooth the close prices. 
    Optionally visualizes the original and smoothed close prices.
    """
    symbol_settings = settings.get(symbol)
        
    # Extract settings for the symbol
    polyorder = symbol_settings["polyorder"]
    window_length = symbol_settings["window_length"]
    
    # Ensure df["Close"] is treated as a numpy array for efficiency
    close_prices = df["Close"].to_numpy()
    
    # Apply the Savitzky-Golay filter
    smoothed_close = savgol_filter(close_prices, window_length, polyorder)
    
    # Add smoothed close prices to DataFrame
    df["smoothed_close"] = smoothed_close
    
    print(f'P: {polyorder}')
    print(f'W: {window_length}')
    
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, close_prices, label="Close Price")
        plt.plot(df.index, smoothed_close, label="Smoothed Close Price")
        plt.legend()
        plt.show()

def detect_pivot_points(df, symbol, visualize=False):
    """
    Detect pivot points in the smoothed close prices. Optionally visualizes the detected pivots.
    """

    order = settings[symbol]["order"]
    print(f'O: {order}')
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

def initialize_sent_limits():
    """
    Initialize or reset the sent limits database. Ensures the file exists with proper headers.
    """
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
    """
    Perform backtesting on historical data for a given symbol. This function simulates trades based on the provided parameters and calculates performance metrics such as wins, losses, and balance changes over time.
    
    Parameters:
    - df: DataFrame containing historical OHLCV data.
    - plot_df: DataFrame containing trade signals generated by a strategy.
    - RR: Risk/Reward ratio for trades.
    - initial_balance: Starting balance for the backtest.
    - risk_amount: Amount of money at risk per trade.
    - risk_type: Type of risk management ('percentage' or 'fixed').
    - symbol: Trading pair symbol to backtest.
    
    Returns:
    - A DataFrame summarizing the backtest results, including entry price, stop loss, take profit, and balance after each trade.
    - Number of wins, losses, and neither (running or pending) trades.
    - Final balance after completing all trades.
    """
    live_trading = True

    try:
        if live_trading:
            balance = mt5.account_info().balance
        else:
            balance = initial_balance
    except Exception as e:
        print(f"An error occurred while getting account balance: {e}")
        return pd.DataFrame(), 0, 0, 0

    results = []
    balance_history = []

    sent_limits = pd.read_csv(order_db) if os.path.exists(order_db) else pd.DataFrame(columns=["symbol", "entry_price"])

    entries = plot_df['Entry'].values
    stop_losses = plot_df['Stop_Loss'].values
    take_profits = plot_df['Take_Profit'].values
    occurrences = plot_df['Occurence'].values

    high_prices = df['High'].values
    low_prices = df['Low'].values

    wins = 0
    losses = 0
    neither = 0

    for idx, entry_price in enumerate(entries):
        stop_loss = stop_losses[idx]
        take_profit = take_profits[idx]
        occurrence_time = occurrences[idx]

        if live_trading and not sent_limits[(sent_limits['symbol'] == symbol) & (sent_limits['entry_price'].astype(float) == entry_price)].empty:
            continue

        occurrence_index = df.index.get_loc(occurrence_time)
        entry_reached = False

        if risk_type == "percentage":
            Risk = risk_amount / 100 * balance
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
    """
    Analyze a single trading symbol across various timeframes using a predefined strategy. This function performs historical data retrieval, preprocessing, and backtesting, then aggregates the results for reporting.
    
    Parameters:
    - symbol: Trading pair symbol to analyze.
    """
    summary_results = []
    strategies = ["Noir"]

    for timeframe_name, timeframe in timeframes.items():
        

        get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)

        # get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)
        df = prep_data(symbol, timeframe_name, visualize=False)
        clean_data(df, symbol)
        detect_pivot_points(df, symbol)

        strategy_results = []

        for strategy_name in strategies:
            strategy = Strategy(df)
            plot_df = getattr(strategy, strategy_name)(RR=5)

            initial_balance = mt5.account_info().balance
            risk_amount = 25
            risk_type = "fixed"
            backtest_results_df, wins, losses, neither = backtest(df, plot_df, RR=5, initial_balance=initial_balance, risk_amount=risk_amount, risk_type=risk_type, symbol=symbol)

            final_balance = backtest_results_df.iloc[-1]["Balance"]
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) != 0 else 0

            consecutive_sl = np.diff((backtest_results_df["Result"] == "SL").astype(int)).cumsum().max()

            strategy_results.append({
                "Symbol": symbol,
                "Timeframe": timeframe_name,
                "Strategy": strategy_name,
                "Wins": wins,
                "Losses": losses,
                "Neither": neither,
                "Consecutive SL": consecutive_sl,
                "Final Balance": final_balance,
                "Win Rate": win_rate,
            })

        summary_results.extend(strategy_results)

        summary_df = pd.DataFrame(summary_results)
        print(summary_df)
        summary_df.to_csv(os.path.join(backtest_summary_dir, "live_trading_summary.csv"), index=False)

def main():
    """
    Main function to orchestrate the execution of the trading bot. It initializes MetaTrader5, sets up the environment, and runs the analysis for each trading symbol.
    """

if __name__ == "__main__":
    initialize_mt5()
    initialize_sent_limits()
    while True:
        for symbol in trading_pairs.symbols:
            for timeframe_name, timeframe in timeframes.items():
                print(f"Running live trading f❤️  r {symbol} 😊 n {timeframe_name} timeframe...")
                analyze_symbol(symbol)
        sleep_time = 0.1
        sleep_time = sleep_time * 60
        print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)