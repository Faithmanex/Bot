# Main trading bot script
import MetaTrader5 as mt5
import config
import pandas as pd
import numpy as np
import time
import logging

# --- Logger Setup ---
def setup_logger():
    """Sets up the root logger for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set global minimum log level

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler('trading_bot.log')
    file_handler.setLevel(logging.INFO)  # Log INFO and above to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Log INFO and above to console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Helper Functions ---
def get_symbol_point_value(symbol):
    """
    Fetches symbol information and returns the 'point' size.
    'point' is the smallest price unit. SL/TP PIPS in this bot are multiples of this value.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}. Error code: {mt5.last_error()}")
        return None
    return symbol_info.point

# --- Trading Functions ---
def open_trade(action, symbol, lot_size, stop_loss_pips, take_profit_pips):
    """
    Opens a trade (BUY or SELL) with specified parameters.
    Note: stop_loss_pips and take_profit_pips are interpreted as multiples of the symbol's 'point' value.
    """
    order_type = None
    if action == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
    elif action == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
    else:
        logging.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol} to open trade. Error code: {mt5.last_error()}")
        return None

    point = symbol_info.point
    price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    if action == "BUY":
        price = symbol_info.ask
        sl_price = price - stop_loss_pips * point
        tp_price = price + take_profit_pips * point
    else: # SELL
        price = symbol_info.bid
        sl_price = price + stop_loss_pips * point
        tp_price = price - take_profit_pips * point
    
    if price == 0: # Price not available, symbol not trading or not in MarketWatch
        logging.warning(f"Price for {symbol} is currently 0.0. Check MarketWatch or symbol properties.")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": round(sl_price, symbol_info.digits),
        "tp": round(tp_price, symbol_info.digits),
        "deviation": 10,  # Allowable price deviation in points
        "magic": 234000,  # Example magic number
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate Or Cancel
    }

    logging.info(f"Attempting to {action} {lot_size} of {symbol} at {price} (SL: {request['sl']}, TP: {request['tp']})")
    result = mt5.order_send(request)

    if result is None:
        logging.error(f"Order send failed for {symbol}. No result object returned. Last error: {mt5.last_error()}")
        return None
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order send failed for {symbol}. Retcode: {result.retcode} - {result.comment}")
        # Additional details for common errors
        if result.retcode == mt5.TRADE_RETCODE_CONNECTION:
             logging.error("  Error: Connection issues with the trade server.")
        elif result.retcode == mt5.TRADE_RETCODE_TIMEOUT:
             logging.error("  Error: Order timed out.")
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS:
             logging.error("  Error: Invalid SL/TP levels. Check distances from current price and symbol's stop level.")
        elif result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
             logging.error("  Error: Not enough money to execute the order.")
        return result # Return the result object for further inspection if needed

    logging.info(f"{action} order for {symbol} successful. Order ID: {result.order}")
    # time.sleep(1) # Optional pause
    return result

def close_trade(position_ticket, lot_size_to_close):
    """
    Closes a trade for a given position ticket.
    `lot_size_to_close` specifies the volume to close.
    """
    positions = mt5.positions_get(ticket=position_ticket)
    if positions is None or len(positions) == 0:
        logging.warning(f"No position found with ticket {position_ticket} to close. Error code: {mt5.last_error()}")
        return None
    if len(positions) > 1:
        logging.warning(f"Multiple positions found with ticket {position_ticket}. This should not happen. Cannot close.")
        return None # Or handle as appropriate

    position = positions[0]
    pos_symbol = position.symbol
    pos_volume = position.volume
    original_order_type = position.type # This is mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL

    # If lot_size_to_close is greater than position volume, adjust to position volume for full close
    if lot_size_to_close > pos_volume:
        logging.info(f"Lot size to close ({lot_size_to_close}) is greater than position volume ({pos_volume}). Adjusting to close full position.")
        lot_size_to_close = pos_volume
    
    symbol_info = mt5.symbol_info(pos_symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {pos_symbol} to close trade. Error code: {mt5.last_error()}")
        return None

    close_order_type = None
    price = 0.0

    if original_order_type == mt5.ORDER_TYPE_BUY: # Original was BUY, so we SELL to close
        close_order_type = mt5.ORDER_TYPE_SELL
        price = symbol_info.bid
    elif original_order_type == mt5.ORDER_TYPE_SELL: # Original was SELL, so we BUY to close
        close_order_type = mt5.ORDER_TYPE_BUY
        price = symbol_info.ask
    else:
        logging.error(f"Unknown original order type: {original_order_type} for position {position_ticket}")
        return None

    if price == 0: # Price not available
        logging.warning(f"Price for {pos_symbol} is currently 0.0 for closing. Check MarketWatch.")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos_symbol,
        "volume": lot_size_to_close,
        "type": close_order_type,
        "position": position_ticket,
        "price": price,
        "deviation": 10,
        "magic": 234001, # Can be same or different from open magic
        "comment": "python script close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logging.info(f"Attempting to close {lot_size_to_close} of position ticket {position_ticket} for {pos_symbol} at {price}")
    result = mt5.order_send(request)

    if result is None:
        logging.error(f"Close order failed for position {position_ticket}. No result object. Last error: {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Close order failed for position {position_ticket}. Retcode: {result.retcode} - {result.comment}")
        return result
    
    logging.info(f"Close order for position {position_ticket} successful. Order ID: {result.order}")
    # time.sleep(1) # Optional pause
    return result


def get_price_data(symbol, timeframe_mt5_constant, count):
    """Fetches historical price data from MT5 and returns it as a pandas DataFrame."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5_constant, 0, count)
        if rates is None:
            logging.error(f"Failed to retrieve price data for {symbol}. Error code: {mt5.last_error()}")
            return None
        if len(rates) == 0:
            logging.warning(f"No price data retrieved for {symbol} (empty rates).")
            return None

        df = pd.DataFrame(rates)
        # Convert time in seconds to datetime objects
        df['time'] = pd.to_datetime(df['time'], unit='s')
        logging.info(f"Price data for {symbol} fetched successfully. Shape: {df.shape}")
        # logging.debug(df.head())
        return df
    except Exception as e:
        logging.exception(f"Error in get_price_data for {symbol}:")
        return None

def calculate_moving_averages(df, short_period, long_period):
    """Calculates short and long Simple Moving Averages (SMAs)."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty or None, cannot calculate MAs.")
        return None
    try:
        df['short_ma'] = df['close'].rolling(window=short_period).mean()
        df['long_ma'] = df['close'].rolling(window=long_period).mean()
        logging.info(f"Moving averages calculated: short_ma ({short_period}), long_ma ({long_period}).")
        # logging.debug(df[['time', 'close', 'short_ma', 'long_ma']].tail())
        return df
    except Exception as e:
        logging.exception("Error in calculate_moving_averages:")
        return None

def check_signal(df):
    """Checks for a trading signal based on MA crossover."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty or None, cannot check signal.")
        return "HOLD" # Or None
    
    # Ensure there are enough data points to make a comparison after MA calculation
    # We need at least two valid (non-NaN) MA values. The long_ma will have more NaNs.
    # Check if the last two long_ma values are not NaN
    if df['long_ma'].isna().iloc[-2:].any():
        logging.warning("Not enough data for signal generation (NaNs in recent MAs).")
        # logging.debug(df[['time', 'close', 'short_ma', 'long_ma']].tail())
        return "HOLD" # Or None

    try:
        last_short_ma = df['short_ma'].iloc[-1]
        last_long_ma = df['long_ma'].iloc[-1]
        prev_short_ma = df['short_ma'].iloc[-2]
        prev_long_ma = df['long_ma'].iloc[-2]

        logging.info(f"Signal Check: Time={df['time'].iloc[-1]}")
        logging.info(f"  Last Short MA: {last_short_ma:.5f}, Last Long MA: {last_long_ma:.5f}")
        logging.info(f"  Prev Short MA: {prev_short_ma:.5f}, Prev Long MA: {prev_long_ma:.5f}")

        if last_short_ma > last_long_ma and prev_short_ma < prev_long_ma:
            logging.info("  Signal: BUY")
            return "BUY"
        elif last_short_ma < last_long_ma and prev_short_ma > prev_long_ma:
            logging.info("  Signal: SELL")
            return "SELL"
        else:
            logging.info("  Signal: HOLD")
            return "HOLD"
    except IndexError:
        logging.warning("IndexError in check_signal: Not enough data points for comparison (less than 2).")
        return "HOLD" # Or None
    except Exception as e:
        logging.exception("Error in check_signal:")
        return "HOLD" # Or None

def initialize_mt5():
    """Initializes the MetaTrader 5 terminal and logs in."""
    try:
        # Initialize MetaTrader 5
        if not mt5.initialize():
            logging.error(f"Failed to initialize MetaTrader 5. Error code: {mt5.last_error()}")
            return False
        logging.info("MetaTrader 5 initialized successfully.")

        # Check if config attributes for login are set
        if not hasattr(config, 'MT5_LOGIN') or \
           not hasattr(config, 'MT5_PASSWORD') or \
           not hasattr(config, 'MT5_SERVER'):
            logging.error("MT5 connection details (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER) are not set in config.py.")
            logging.error("Please ensure these are uncommented and correctly set in your config.py file.")
            # mt5.shutdown() # Optional: shutdown if config is missing, or let it try and fail at login
            return False # Or handle as a specific type of failure

        # Login to the account
        logging.info(f"Attempting to login with Login ID: {config.MT5_LOGIN} to Server: {config.MT5_SERVER}")
        authorized = mt5.login(login=config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER)
        if authorized:
            logging.info("Logged in to MetaTrader 5 account successfully.")
            account_info = mt5.account_info()
            if account_info is not None:
                logging.info("Account Info:")
                logging.info(f"  Login: {account_info.login}")
                logging.info(f"  Name: {account_info.name}")
                logging.info(f"  Server: {account_info.server}")
                logging.info(f"  Balance: {account_info.balance} {account_info.currency}")
            else:
                logging.warning("Failed to retrieve account information.")
            return True
        else:
            logging.error(f"Failed to log in to MetaTrader 5 account. Error code: {mt5.last_error()}")
            mt5.shutdown() # Shutdown if login failed
            return False

    except AttributeError as e:
        logging.exception(f"Configuration error: {e}. Make sure MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER are defined in config.py.")
        # It's good practice to also shutdown MT5 if it was initialized before the error
        if mt5.terminal_state() is not None and mt5.terminal_state().connected:
             mt5.shutdown()
        return False
    except Exception as e:
        logging.exception("An error occurred during MT5 initialization or login:")
        # It's good practice to also shutdown MT5 if it was initialized before the error
        if mt5.terminal_state() is not None and mt5.terminal_state().connected:
             mt5.shutdown()
        return False

def shutdown_mt5():
    """Shuts down the MetaTrader 5 terminal connection."""
    logging.info("Shutting down MetaTrader 5 connection...")
    mt5.shutdown()
    logging.info("MetaTrader 5 connection shut down.")

# --- Main Bot Loop ---
def main_loop():
    """Main operational loop for the trading bot."""
    logging.info("Starting main trading loop...")
    while True:
        logging.info(f"--- New Iteration at {pd.Timestamp.now(tz='UTC')} ---")
        signal = "HOLD" # Default signal

        try:
            # Fetch Data and Signal
            price_data = get_price_data(config.SYMBOL, config.TIMEFRAME_MT5, count=config.LONG_MA_PERIOD + 50)

            if price_data is not None and not price_data.empty:
                price_data_with_ma = calculate_moving_averages(price_data, config.SHORT_MA_PERIOD, config.LONG_MA_PERIOD)
                if price_data_with_ma is not None and not price_data_with_ma.empty:
                    signal = check_signal(price_data_with_ma)
                    logging.info(f"Current signal for {config.SYMBOL}: {signal}")
                else:
                    logging.warning("Failed to calculate moving averages. Holding.")
                    signal = "HOLD" # Ensure signal is HOLD if MAs fail
            else:
                logging.warning("Failed to retrieve price data. Retrying sooner.")
                # Shorter sleep if data retrieval fails, then continue to next iteration
                time.sleep(config.LOOP_SLEEP_SECONDS / 2 if hasattr(config, 'LOOP_SLEEP_SECONDS') else 30)
                continue

            # Trading Logic
            if signal in ["BUY", "SELL"]:
                positions = mt5.positions_get(symbol=config.SYMBOL)
                if positions is None:
                    logging.error("Error checking positions. Possible connection issue. Holding.")
                elif len(positions) == 0: # No open position for this symbol
                    action_msg = f"Action: Attempting to open {signal} trade for {config.SYMBOL}"
                    logging.info(action_msg)
                    open_trade(
                        action=signal,
                        symbol=config.SYMBOL,
                        lot_size=config.LOT_SIZE,
                        stop_loss_pips=config.STOP_LOSS_PIPS,
                        take_profit_pips=config.TAKE_PROFIT_PIPS
                    )
                else: # There is an open position for this symbol
                    current_position = positions[0]
                    pos_type = "BUY" if current_position.type == mt5.ORDER_TYPE_BUY else "SELL"
                    logging.info(f"Action: Hold. Position already open for {config.SYMBOL}: Type: {pos_type}, Ticket: {current_position.ticket}, Price: {current_position.price_open}, Volume: {current_position.volume}")
                    # Optional: If signal opposes pos_type, consider closing. For now, just hold.
                    # if (signal == "BUY" and pos_type == "SELL") or \
                    #    (signal == "SELL" and pos_type == "BUY"):
                    #     logging.info(f"Opposing signal ({signal}) to current {pos_type} position. Consider closing logic here.")
                    #     # close_trade(current_position.ticket, current_position.volume)
            else: # Signal is "HOLD" or invalid
                logging.info(f"Action: Hold. No new trade signal ({signal}) for {config.SYMBOL}.")

        except Exception as e:
            logging.exception("An unexpected error occurred in the main loop:")
            # Depending on the error, you might want to sleep longer or attempt to re-initialize
            # For now, just print and continue with the normal sleep cycle

        # Sleep
        sleep_duration = config.LOOP_SLEEP_SECONDS if hasattr(config, 'LOOP_SLEEP_SECONDS') else 60
        logging.info(f"Waiting for {sleep_duration} seconds...")
        time.sleep(sleep_duration)


if __name__ == "__main__":
    setup_logger() # Setup logger first
    logging.info("Bot starting...")
    if initialize_mt5():
        try:
            main_loop()
        except KeyboardInterrupt:
            logging.info("Bot shutting down due to user request (Ctrl+C).")
        except Exception as e:
            logging.exception("Critical error in bot execution:")
        finally:
            shutdown_mt5()
    else:
        logging.error("Failed to initialize MT5. Bot cannot start.")
    logging.info("Bot has stopped.")
