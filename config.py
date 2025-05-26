import MetaTrader5 as mt5

# MT5 Connection Settings (replace with your demo account credentials)
# MT5_LOGIN = 1234567
# MT5_PASSWORD = "your_password"
# MT5_SERVER = "Your_Broker_Server"

# Trading Parameters
SYMBOL = "EURUSD"
TIMEFRAME_MT5 = mt5.TIMEFRAME_H1 # MetaTrader 5 timeframe, e.g., M1, M5, M15, M30, H1, H4, D1, W1, MN1

# Strategy Parameters (example for Moving Average Crossover)
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 50

# Risk Management
LOT_SIZE = 0.01
STOP_LOSS_PIPS = 50 # Stop loss in pips. Note: In this bot, PIPS are treated as multiples of symbol's 'point'.
TAKE_PROFIT_PIPS = 100 # Take profit in pips. Note: In this bot, PIPS are treated as multiples of symbol's 'point'.

# Bot Operation
LOOP_SLEEP_SECONDS = 60 # Time in seconds between each check in the main loop
