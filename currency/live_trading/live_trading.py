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
    
    col_map = {col.lower(): col for col in df.columns}
    open_col = col_map.get("open", "open")
    high_col = col_map.get("high", "high")
    low_col = col_map.get("low", "low")
    close_col = col_map.get("close", "close")
    vol_col = col_map.get("tick_volume", col_map.get("volume", "tick_volume"))
    spread_col = col_map.get("spread", "spread")
    
    ohlcv_df = pd.DataFrame(index=df.index)
    ohlcv_df["Open"] = df[open_col]
    ohlcv_df["High"] = df[high_col]
    ohlcv_df["Low"] = df[low_col]
    ohlcv_df["Close"] = df[close_col]
    ohlcv_df["Volume"] = df[vol_col]
    if spread_col in df.columns:
        ohlcv_df["spread"] = df[spread_col]
    else:
        ohlcv_df["spread"] = 0.0
    
    if visualize:
        mpf.plot(ohlcv_df[["Open", "High", "Low", "Close", "Volume"]], type="candle", style="line", title=f"{symbol} {timeframe_name}", volume=True)
    
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
    Initialize sent limits by checking MetaTrader 5 active orders directly.
    """
    print("Initializing MetaTrader 5 active orders tracking...")


def get_symbol_min_stop_distance(symbol, sample_price=None):
    """
    Get the minimum stop loss/take profit distance in price for the symbol.
    Strictly queries MT5 online. Raises RuntimeError if offline.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"[ERROR] Symbol info for {symbol} could not be retrieved from MetaTrader 5. The bot must be fully online.")
    stops_level = symbol_info.trade_stops_level
    point = symbol_info.point
    return stops_level * point


def _place_pending_order(symbol, entry_price, stop_loss, take_profit, risk_amount, risk_type, balance):
    try:
        from lot_size import get_lot_size
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
            "type": mt5.ORDER_TYPE_BUY_LIMIT if stop_loss < entry_price else mt5.ORDER_TYPE_SELL_LIMIT,
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
        if res is None:
            print(f"Failed to place order for {symbol} @ {entry_price}: no response from MT5.")
            return
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place order for {symbol} @ {entry_price}: retcode={res.retcode}")
    except Exception as e:
        print(f"[ERROR] Exception during order placement for {symbol} @ {entry_price}: {e}")


def _reconcile_pending_orders(symbol, desired_pending_patterns, risk_amount, risk_type, balance):
    """
    Reconciles pending limit orders on MetaTrader 5 with the desired recent patterns.
    Implements a FIFO queue logic where new patterns displace oldest active ones,
    while empty slots are filled from the oldest inactive stored pending patterns.
    Everything is adjustable live, and TP/SL are checked to ensure active orders stay up to date.
    """
    # 1. Reload global settings dynamically to pick up live adjustments
    live_settings = load_settings()
    
    # 2. Get live customizable limits from settings
    # Default: max 5 active pending orders per symbol
    max_pending = live_settings.get("max_pending_orders", 5)
    # Default: max age of 48 hours for a pending pattern
    max_age_hours = live_settings.get("max_pending_order_age_hours", 48)

    # Check per-symbol overrides if present
    sym_cfg = live_settings.get(symbol, {})
    if isinstance(sym_cfg, dict):
        max_pending = sym_cfg.get("max_pending_orders", max_pending)
        max_age_hours = sym_cfg.get("max_pending_order_age_hours", max_age_hours)

    print(f"[RECONCILE] Running pending order reconciliation for {symbol} (Limit: {max_pending} slots, Max Age: {max_age_hours}h)...")

    # 3. Filter desired patterns to keep only those within the allowed max age
    now = datetime.now()
    recent_patterns = []
    for p in desired_pending_patterns:
        occ = p.get("Occurrence")
        # Ensure occurrence time is a naive datetime object for timezone comparison
        if isinstance(occ, str):
            try:
                occ_dt = pd.to_datetime(occ)
            except Exception:
                occ_dt = now
        else:
            occ_dt = occ
        
        if hasattr(occ_dt, "tzinfo") and occ_dt.tzinfo is not None:
            occ_dt = occ_dt.replace(tzinfo=None)
            
        age = now - occ_dt
        if age <= timedelta(hours=max_age_hours):
            recent_patterns.append(p)

    # Ensure chronological order (oldest to newest)
    recent_patterns.sort(key=lambda x: x.get("Occurrence"))

    # 4. Fetch current pending orders on MetaTrader 5 for this symbol
    active_orders = mt5.orders_get(symbol=symbol)
    if active_orders is None:
        active_orders = []
        
    # Only manage active pending orders placed by this bot
    bot_active_orders = [o for o in active_orders if o.comment == "Echelnet Bot"]

    # 5. Cancel any active order on MT5 that is NO LONGER in our recent_patterns at all
    # (e.g. because it was filled, hit SL/TP, or is older than max_age_hours)
    valid_active_orders = []
    for order in bot_active_orders:
        matched = False
        for p in recent_patterns:
            if abs(order.price_open - float(p["Entry"])) < 1e-5:
                matched = True
                break
        if not matched:
            print(f"[RECONCILE] Cancelling invalidated pending order for {symbol} @ price {order.price_open} (Ticket: {order.ticket})")
            cancel_request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket
            }
            res = mt5.order_send(cancel_request)
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                err_code = mt5.last_error() if hasattr(mt5, 'last_error') else 'N/A'
                print(f"[RECONCILE] Failed to cancel order {order.ticket}: {err_code}")
        else:
            valid_active_orders.append(order)

    # 6. Check if TP and SL are up-to-date for the remaining active orders
    # If not, cancel the outdated active order so it will be replaced with updated TP/SL!
    up_to_date_active_orders = []
    for order in valid_active_orders:
        tp_sl_correct = False
        for p in recent_patterns:
            if abs(order.price_open - float(p["Entry"])) < 1e-5:
                # Check stop loss and take profit (floating point check within a small epsilon)
                sl_match = abs(order.sl - float(p["Stop_Loss"])) < 1e-5
                tp_match = abs(order.tp - float(p["Take_Profit"])) < 1e-5
                if sl_match and tp_match:
                    tp_sl_correct = True
                break
        
        if not tp_sl_correct:
            print(f"[RECONCILE] Cancelling pending order with outdated TP/SL for {symbol} @ price {order.price_open} (Ticket: {order.ticket})")
            cancel_request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket
            }
            res = mt5.order_send(cancel_request)
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                err_code = mt5.last_error() if hasattr(mt5, 'last_error') else 'N/A'
                print(f"[RECONCILE] Failed to cancel outdated order {order.ticket}: {err_code}")
        else:
            up_to_date_active_orders.append(order)

    # Update active count (K) based on the remaining up-to-date active orders
    K = len(up_to_date_active_orders)

    # 7. Check if there is a brand-new pattern in our recent list
    # A pattern is "brand-new" if it occurred within the last 15 minutes and is not currently active on MT5
    new_pattern_to_place = None
    if recent_patterns:
        newest_p = recent_patterns[-1]
        occ_dt = newest_p.get("Occurrence")
        if hasattr(occ_dt, "tzinfo") and occ_dt.tzinfo is not None:
            occ_dt = occ_dt.replace(tzinfo=None)
        
        # Check if the newest pattern occurred within the last 15 minutes
        if now - occ_dt <= timedelta(minutes=15):
            # Check if it is not already active
            is_active = False
            for order in up_to_date_active_orders:
                if abs(order.price_open - float(newest_p["Entry"])) < 1e-5:
                    is_active = True
                    break
            if not is_active:
                new_pattern_to_place = newest_p

    # 8. Place the brand-new pattern (if any) with displacement if slots are exceeded
    if new_pattern_to_place:
        entry_price = float(new_pattern_to_place["Entry"])
        stop_loss = float(new_pattern_to_place["Stop_Loss"])
        take_profit = float(new_pattern_to_place["Take_Profit"])
        
        if K >= max_pending:
            # Slots exceeded: find the oldest active order on MT5 that is in recent_patterns
            # We want to cancel/displace it to make room!
            oldest_active_order = None
            for p in recent_patterns:
                # Iterate chronologically (oldest pattern first)
                for order in up_to_date_active_orders:
                    if abs(order.price_open - float(p["Entry"])) < 1e-5:
                        oldest_active_order = order
                        break
                if oldest_active_order:
                    break
            
            if oldest_active_order:
                print(f"[RECONCILE] Active slot limit {max_pending} exceeded. "
                      f"Displacing oldest active order for {symbol} @ price {oldest_active_order.price_open} (Ticket: {oldest_active_order.ticket})")
                cancel_request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": oldest_active_order.ticket
                }
                res = mt5.order_send(cancel_request)
                if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                    # Update active orders list and count
                    up_to_date_active_orders = [o for o in up_to_date_active_orders if o.ticket != oldest_active_order.ticket]
                    K = len(up_to_date_active_orders)
                else:
                    err_code = mt5.last_error() if hasattr(mt5, 'last_error') else 'N/A'
                    print(f"[RECONCILE] Failed to displace oldest active order: {err_code}")
        
        # Place the new pattern!
        if K < max_pending:
            print(f"[RECONCILE] Placing brand-new pending order for {symbol} @ price {entry_price}")
            _place_pending_order(symbol, entry_price, stop_loss, take_profit, risk_amount, risk_type, balance)
            
            # Re-fetch active orders to ensure up_to_date_active_orders list is fresh
            active_orders = mt5.orders_get(symbol=symbol)
            if active_orders is None:
                active_orders = []
            up_to_date_active_orders = [o for o in active_orders if o.comment == "Echelnet Bot"]
            K = len(up_to_date_active_orders)

    # 9. Fill any remaining empty slots starting from the oldest inactive recent patterns
    if K < max_pending:
        for p in recent_patterns:
            entry_price = float(p["Entry"])
            stop_loss = float(p["Stop_Loss"])
            take_profit = float(p["Take_Profit"])
            
            # Check if this pattern is already active on MT5
            already_active = False
            for order in up_to_date_active_orders:
                if abs(order.price_open - entry_price) < 1e-5:
                    already_active = True
                    break
                    
            if not already_active:
                print(f"[RECONCILE] Filling empty slot. Placing stored pending order for {symbol} @ price {entry_price}")
                _place_pending_order(symbol, entry_price, stop_loss, take_profit, risk_amount, risk_type, balance)
                K += 1
                if K >= max_pending:
                    break

    print(f"[RECONCILE] Reconciliation complete. Active pending orders on MT5 for {symbol}: {K}/{max_pending}")


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

    desired_pending_patterns = []

    entries = plot_df['Entry'].values
    stop_losses = plot_df['Stop_Loss'].values
    take_profits = plot_df['Take_Profit'].values
    occurrences = plot_df['Occurence'].values

    high_prices = df['High'].values
    low_prices = df['Low'].values

    import MetaTrader5 as mt5
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"[ERROR] Symbol info for {symbol} could not be retrieved from MetaTrader 5. The bot must be fully online.")
    point_size = info.point
        
    raw_spreads = df['spread'].values if 'spread' in df.columns else np.zeros_like(low_prices)
    spread_prices = raw_spreads * point_size

    wins = 0
    losses = 0
    neither = 0

    min_stop_dist = get_symbol_min_stop_distance(symbol)

    for idx, entry_price in enumerate(entries):
        stop_loss = stop_losses[idx]
        take_profit = take_profits[idx]
        occurrence_time = occurrences[idx]

        # Validate stop loss distance to prevent 'invalid stop loss' broker rejections
        sl_dist = abs(entry_price - stop_loss)
        if sl_dist < min_stop_dist:
            print(f"[INFO] Skipping trade for {symbol} at {occurrence_time}: stop loss too close to entry ({sl_dist:.5f} < min {min_stop_dist:.5f})")
            continue

        occurrence_index = df.index.get_loc(occurrence_time)
        entry_reached = False

        if risk_type == "percentage":
            Risk = risk_amount / 100 * balance
        else:
            Risk = risk_amount

        subsequent_highs = high_prices[occurrence_index + 1:]
        subsequent_lows = low_prices[occurrence_index + 1:]
        subsequent_spreads = spread_prices[occurrence_index + 1:]

        is_buy = stop_loss < entry_price

        if is_buy:
            entry_reached_mask = (subsequent_lows + subsequent_spreads) <= entry_price
        else:
            entry_reached_mask = subsequent_highs >= entry_price

        if np.any(entry_reached_mask):
            entry_reached = True
            first_entry_index = np.argmax(entry_reached_mask)
            
            post_highs = subsequent_highs[first_entry_index:]
            post_lows = subsequent_lows[first_entry_index:]
            post_spreads = subsequent_spreads[first_entry_index:]

            if is_buy:
                stop_loss_reached_mask = post_lows <= stop_loss
                take_profit_reached_mask = post_highs >= take_profit
            else:
                stop_loss_reached_mask = (post_highs + post_spreads) >= stop_loss
                take_profit_reached_mask = (post_lows + post_spreads) <= take_profit

            if np.any(stop_loss_reached_mask):
                stop_loss_reached_index = np.argmax(stop_loss_reached_mask)
            else:
                stop_loss_reached_index = len(post_highs)

            if np.any(take_profit_reached_mask):
                take_profit_reached_index = np.argmax(take_profit_reached_mask)
            else:
                take_profit_reached_index = len(post_lows)

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
            desired_pending_patterns.append({
                "Occurrence": occurrence_time,
                "Entry": entry_price,
                "Stop_Loss": stop_loss,
                "Take_Profit": take_profit,
            })

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

    if live_trading:
        _reconcile_pending_orders(symbol, desired_pending_patterns, risk_amount, risk_type, balance)

    return pd.DataFrame(results), wins, losses, neither


def analyze_symbol(symbol):
    """
    Analyze a single trading symbol across various timeframes using a predefined strategy. This function performs historical data retrieval, preprocessing, and backtesting, then aggregates the results for reporting.
    
    Parameters:
    - symbol: Trading pair symbol to analyze.
    """
    global settings
    settings = load_settings()

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
    # Lazily resolve broker symbols after MT5 connection is established
    trading_pairs.symbols = trading_pairs.get_matched_symbols(trading_pairs.symbols)
    while True:
        for symbol in trading_pairs.symbols:
            for timeframe_name, timeframe in timeframes.items():
                print(f"Running live trading f❤️  r {symbol} 😊 n {timeframe_name} timeframe...")
                analyze_symbol(symbol)
        sleep_time = 0.1
        sleep_time = sleep_time * 60
        print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)