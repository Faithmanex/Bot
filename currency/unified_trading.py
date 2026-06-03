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
    rates = None
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    except Exception:
        pass

    if rates is None or len(rates) == 0:
        # Check if local CSV file already exists for offline/no-internet mode
        filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
        if os.path.exists(filename):
            print(f"[INFO] MT5 copy_rates failed (possible offline mode). Falling back to local CSV for {symbol}.")
            return True
        print(f"No data retrieved for {symbol}, error code = {mt5.last_error() if hasattr(mt5, 'last_error') else 'N/A'}")
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
    
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if "spread" in df.columns:
        cols.append("spread")
    ohlcv_df = df[cols].copy()
    if "spread" not in ohlcv_df.columns:
        ohlcv_df["spread"] = 0.0
        
    if visualize:
        mpf.plot(ohlcv_df[["Open", "High", "Low", "Close", "Volume"]], type="candle", style="line", title=f"{symbol} {timeframe_name}", volume=True)
    return ohlcv_df


def clean_data(df, symbol, timeframe=None, visualize=False):
    model_symbol = f"{symbol}_{timeframe}" if timeframe else symbol
    sym_cfg = settings.get(model_symbol, settings.get(symbol, {}))
    polyorder     = sym_cfg.get("polyorder",     6)
    window_length = sym_cfg.get("window_length", 7)
    close_prices = df["Close"].to_numpy()
    smoothed_close = savgol_filter(close_prices, window_length, polyorder)
    df["smoothed_close"] = smoothed_close
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, close_prices, label="Close Price")
        plt.plot(df.index, smoothed_close, label="Smoothed Close Price")
        plt.legend()
        plt.show()


def detect_pivot_points(df, symbol, timeframe=None, visualize=False):
    model_symbol = f"{symbol}_{timeframe}" if timeframe else symbol
    sym_cfg = settings.get(model_symbol, settings.get(symbol, {}))
    order = sym_cfg.get("order", 5)
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


def run_strategy(df, plot_df, RR, initial_balance, risk_amount, risk_type, symbol, live_trading=False):
    # Try querying account info. If MT5 is offline/uninitialized, fall back to initial_balance
    balance = initial_balance
    if live_trading:
        try:
            balance = mt5.account_info().balance
        except Exception:
            pass

    wins, losses, neither = 0, 0, 0
    results = []
    balance_history = []

    desired_pending_patterns = []

    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"[ERROR] Symbol info for {symbol} could not be retrieved from MetaTrader 5. The bot must be fully online.")
    point_size = symbol_info.point
        
    raw_spreads = df["spread"].to_numpy() if "spread" in df.columns else np.zeros_like(low_arr)
    spread_arr = raw_spreads * point_size
    index_arr = df.index
 
    min_stop_dist = get_symbol_min_stop_distance(symbol)

    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence

        # Validate stop loss distance to prevent 'invalid stop loss' broker rejections
        sl_dist = abs(entry_price - stop_loss)
        if sl_dist < min_stop_dist:
            print(f"[INFO] Skipping trade for {symbol} at {occurrence_time}: stop loss too close to entry ({sl_dist:.5f} < min {min_stop_dist:.5f})")
            continue

        try:
            occ_loc = index_arr.get_loc(occurrence_time)
        except KeyError:
            neither += 1
            continue

        # Vectorized: search for entry, SL, TP after occurrence
        future_high = high_arr[occ_loc + 1:]
        future_low = low_arr[occ_loc + 1:]
        future_spread = spread_arr[occ_loc + 1:]
        future_index = index_arr[occ_loc + 1:]

        is_buy = stop_loss < entry_price

        if is_buy:
            entry_mask = (future_low + future_spread) <= entry_price
        else:
            entry_mask = future_high >= entry_price

        if not entry_mask.any():
            neither += 1
            if live_trading:
                desired_pending_patterns.append({
                    "Occurrence": occurrence_time,
                    "Entry": entry_price,
                    "Stop_Loss": stop_loss,
                    "Take_Profit": take_profit,
                })
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
        post_spread = future_spread[entry_pos:]

        if is_buy:
            sl_mask = post_low <= stop_loss
            tp_mask = post_high >= take_profit
        else:
            sl_mask = (post_high + post_spread) >= stop_loss
            tp_mask = (post_low + post_spread) <= take_profit

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

    if live_trading:
        _reconcile_pending_orders(symbol, desired_pending_patterns, risk_amount, risk_type, balance)

    return pd.DataFrame(results), wins, losses, neither


def _place_pending_order(symbol, entry_price, stop_loss, take_profit, risk_amount, risk_type, balance):
    try:
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


def analyze_symbol(symbol, live_trading=False, config=None):
    global settings
    settings = load_settings()

    if config is None:
        config = {}

    summary_results = []
    strategies = [config.get("strategy", "Noir")]
    timeframe_name = config.get("timeframe", "M5")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M10": mt5.TIMEFRAME_M10,
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
    clean_data(df, symbol, timeframe=timeframe_name)
    detect_pivot_points(df, symbol, timeframe=timeframe_name)

    for strategy_name in strategies:
        if strategy_name == "MLPattern":
            from .modules.ml_pattern import build_and_train_model, MODEL_DIR
            model_filename = f"{symbol}_{timeframe_name}_pattern_model.joblib" if timeframe_name else f"{symbol}_pattern_model.joblib"
            model_path = os.path.join(MODEL_DIR, model_filename)
            if not os.path.exists(model_path):
                print(f"[INFO] Saved model for {symbol}_{timeframe_name} not found. Building and training ML model first...")
                build_and_train_model(df, symbol, RR=config.get("rr", 5.0), timeframe=timeframe_name)
        
        strategy = Strategy(df, symbol=symbol, timeframe=timeframe_name)
        rr = config.get("rr", 5.0)
        plot_df = getattr(strategy, strategy_name)(RR=rr)

        # Filter allowed setup direction (Both, Buys Only, Sells Only)
        direction = config.get("direction", "Both")
        if not plot_df.empty:
            if direction == "Buys Only":
                plot_df = plot_df[plot_df["Stop_Loss"] < plot_df["Entry"]].copy()
            elif direction == "Sells Only":
                plot_df = plot_df[plot_df["Stop_Loss"] > plot_df["Entry"]].copy()

        initial_balance = config.get("initial_balance", 1000.0)
        risk_amount = config.get("risk_amount", 25.0)
        risk_type = config.get("risk_type", "fixed")

        backtest_results_df, wins, losses, neither = run_strategy(
            df, plot_df, RR=rr, initial_balance=initial_balance,
            risk_amount=risk_amount, risk_type=risk_type,
            symbol=symbol, live_trading=live_trading,
        )

        if not backtest_results_df.empty:
            # Save detailed trade results for plotting the equity curve in GUI
            detailed_filename = f"detailed_results_{symbol}.csv"
            backtest_results_df.to_csv(os.path.join(BACKTEST_SUMMARY_DIR, detailed_filename), index=False)
            
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

    mt5_ok = initialize_mt5()
    if live_trading and not mt5_ok:
        print("[ERROR] Cannot run live trading: MetaTrader 5 initialization failed.")
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
        if mt5_ok:
            shutdown_mt5()


if __name__ == "__main__":
    main(live_trading=False)
