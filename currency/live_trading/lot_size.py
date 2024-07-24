import MetaTrader5 as mt5

def get_symbol_properties(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found. Please check the symbol.")
        return None
    return symbol_info

def get_currency_rate(symbol, account_currency, symbol_currency):
    if account_currency == symbol_currency:
        return 1.0  # No conversion needed for the same currency

    tick = mt5.symbol_info_tick(symbol)
    
    if tick is not None:
        return tick.bid
    else:
        print(f"Unable to retrieve tick information for symbol {symbol}")
        return None

def calculate_lot_size(account_balance, risk_amount, stop_loss, entry_price, symbol_properties, account_currency):
    if symbol_properties is None:
        return None

    contract_size = symbol_properties.trade_contract_size
    tick_size = symbol_properties.trade_tick_size
    symbol_currency = symbol_properties.currency_profit

    if symbol_currency != 'USD':
        exchange_rate = get_currency_rate(symbol_properties.name, account_currency, symbol_currency)
    else:
        exchange_rate = 1.0  # No conversion needed for USD

    lot_size = (risk_amount / ((stop_loss - entry_price) * contract_size)) * exchange_rate
    return lot_size

def get_lot_size(risk_amount, stop_loss, account_currency, symbol, risk_type, account_balance, entry_price):
    mt5.initialize()

    symbol_properties = get_symbol_properties(symbol)
    if symbol_properties:
        if risk_type == "percentage":
            risk_amount = (risk_amount / 100) * account_balance
        lot_size = calculate_lot_size(account_balance, risk_amount, stop_loss, entry_price, symbol_properties, account_currency)
        if lot_size is not None:
            return round(lot_size, 2)

    mt5.shutdown()
    return None

# Example usage of the program
# account_balance = 10000
# risk_amount = 20
# entry_price = 6250
# stop_loss = 6280
# symbol = "Volatility 10 (1s) Index"
# risk_type = "percentage"
# account_currency = "USD"

# lot_size = get_lot_size(risk_amount, stop_loss, account_currency, symbol, risk_type, account_balance, entry_price)
# if lot_size is not None:
#     print(f"Calculated lot size for symbol {symbol}: {lot_size}")
# else:
#     print("Unable to calculate lot size.")
