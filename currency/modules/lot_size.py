import MetaTrader5 as mt5


def get_lot_size(risk_amount, stop_loss, account_currency, symbol, risk_type, account_balance, entry_price):
    """
    Calculates the lot size for a trade based on risk parameters.
    Assumes MT5 is already initialized by the caller.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{symbol} not found, cannot calculate lot size. Error: {mt5.last_error()}")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed, error: {mt5.last_error()}")
            return None

    if risk_type == "percentage":
        risk_value = (risk_amount / 100) * account_balance
    else:
        risk_value = risk_amount

    contract_size = symbol_info.trade_contract_size
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value

    pip_size = 0.01 if "JPY" in symbol else 0.0001

    stop_loss_pips = abs(entry_price - stop_loss) / pip_size
    value_per_pip_per_lot = tick_value / tick_size * pip_size

    if value_per_pip_per_lot == 0:
        print(f"Could not determine value per pip for {symbol}. Cannot calculate lot size.")
        return None

    lot_size = risk_value / (stop_loss_pips * value_per_pip_per_lot)

    volume_step = symbol_info.volume_step
    lot_size = round(lot_size / volume_step) * volume_step
    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))

    return lot_size
