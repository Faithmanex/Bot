import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import savgol_filter, argrelextrema
from scipy.stats import qmc
import json
import MetaTrader5 as mt5

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from .strategy import Strategy as OrigStrategy
from ..settings import load_settings, save_settings, get_path, HISTORY_DATA_DIR, OPT_HISTORY_PATH

STRATEGY_PARAM_DEFAULTS = {
    "Noir": {"min_swing_low_ratio": 1.1, "max_swing_low_ratio": 3.0,
             "min_swing_high_ratio": 1.2, "max_swing_high_ratio": 3.0, "sl_mult": 1.1},
    "BreakerBlock": {"min_range_ratio": 2.0, "sl_fib": 0.272},
    "DoubleTop": {"min_range_ratio": 2.0, "sl_fib": 0.272, "tolerance": 0.02},
    "TripleTop": {"min_range_ratio": 2.0, "sl_fib": 0.272},
}

STRATEGY_PARAM_BOUNDS = {
    "Noir": {"min_swing_low_ratio": (1.0, 2.0), "max_swing_low_ratio": (2.0, 5.0),
             "min_swing_high_ratio": (1.0, 2.5), "max_swing_high_ratio": (2.0, 5.0), "sl_mult": (1.0, 2.5)},
    "BreakerBlock": {"min_range_ratio": (1.0, 4.0), "sl_fib": (0.1, 0.6)},
    "DoubleTop": {"min_range_ratio": (1.0, 4.0), "sl_fib": (0.1, 0.6), "tolerance": (0.005, 0.05)},
    "TripleTop": {"min_range_ratio": (1.0, 4.0), "sl_fib": (0.1, 0.6)},
}

class ParameterizedStrategy(OrigStrategy):
    def __init__(self, dataframe, params=None):
        super().__init__(dataframe)
        if params is None:
            params = {}
        self._sp = {**STRATEGY_PARAM_DEFAULTS.get("Noir", {}), **params}

    def configure(self, strategy_name, params):
        merged = STRATEGY_PARAM_DEFAULTS.get(strategy_name, {}).copy()
        merged.update(params)
        self._sp = merged

    def Noir(self, RR):
        sp = self._sp
        min_sl = sp["min_swing_low_ratio"]
        max_sl = sp["max_swing_low_ratio"]
        min_sh = sp["min_swing_high_ratio"]
        max_sh = sp["max_swing_high_ratio"]
        slm = sp["sl_mult"]
        conditions = (
            self.new_df['low_shift_0'].notna() &
            self.new_df['low_shift_2'].notna() &
            self.new_df['low_shift_4'].notna() &
            self.new_df['high_shift_1'].notna() &
            self.new_df['high_shift_3'].notna() &
            self.new_df['high_shift_5'].notna() &
            (self.new_df['low_shift_0'] < self.new_df['low_shift_2']) &
            (self.new_df['low_shift_4'] > self.new_df['low_shift_2']) &
            (self.new_df['high_shift_1'] < self.new_df['high_shift_3']) &
            ((self.new_df['low_shift_2'] - self.new_df['low_shift_0']) > (min_sl * (self.new_df['high_shift_1'] - self.new_df['low_shift_2']))) &
            ((self.new_df['low_shift_2'] - self.new_df['low_shift_0']) < (max_sl * (self.new_df['high_shift_1'] - self.new_df['low_shift_2']))) &
            ((self.new_df['high_shift_3'] - self.new_df['low_shift_2']) > (min_sh * (self.new_df['high_shift_3'] - self.new_df['low_shift_4']))) &
            ((self.new_df['high_shift_3'] - self.new_df['low_shift_2']) < (max_sh * (self.new_df['high_shift_3'] - self.new_df['low_shift_4'])))
        )
        new_df_filtered = self.new_df[conditions]
        occurences, entries, stop_losses, take_profits = [], [], [], []
        if not new_df_filtered.empty:
            occurences = new_df_filtered.index.tolist()
            block_range = new_df_filtered['high_shift_3'] - new_df_filtered['low_shift_4']
            entry_price = new_df_filtered['high_shift_5']
            stop_loss = new_df_filtered['high_shift_5'] + (block_range * slm)
            take_profit = entry_price - ((stop_loss - entry_price) * RR)
            entries = entry_price.tolist()
            stop_losses = stop_loss.tolist()
            take_profits = take_profit.tolist()
        plot_df = pd.DataFrame({"Occurence": occurences, "Entry": entries,
                                "Stop_Loss": stop_losses, "Take_Profit": take_profits})
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"])
        return plot_df

    def BreakerBlock(self, RR):
        sp = self._sp
        min_range = sp["min_range_ratio"]
        sl_fib = sp["sl_fib"]
        conditions = (
            self.new_df['low_shift_0'].notna() &
            self.new_df['low_shift_2'].notna() &
            self.new_df['low_shift_4'].notna() &
            self.new_df['high_shift_1'].notna() &
            self.new_df['high_shift_3'].notna() &
            (self.new_df['low_shift_0'] < self.new_df['high_shift_1']) &
            (self.new_df['low_shift_2'] < self.new_df['high_shift_1']) &
            (self.new_df['high_shift_3'] > self.new_df['low_shift_2']) &
            (self.new_df['low_shift_0'] < self.new_df['low_shift_2']) &
            (self.new_df['high_shift_3'] < self.new_df['high_shift_1']) &
            (self.new_df['low_shift_4'] < self.new_df['low_shift_2']) &
            ((self.new_df['high_shift_1'] - self.new_df['low_shift_0']) > (min_range * (self.new_df['high_shift_1'] - self.new_df['low_shift_2'])))
        )
        new_df_filtered = self.new_df[conditions]
        occurences, entries, stop_losses, take_profits = [], [], [], []
        if not new_df_filtered.empty:
            occurences = new_df_filtered.index.tolist()
            block_range = new_df_filtered['high_shift_3'] - new_df_filtered['low_shift_2']
            entry_price = new_df_filtered['low_shift_2']
            stop_loss = new_df_filtered['high_shift_1'] + (block_range * sl_fib)
            take_profit = entry_price - ((stop_loss - entry_price) * RR)
            entries = entry_price.tolist()
            stop_losses = stop_loss.tolist()
            take_profits = take_profit.tolist()
        plot_df = pd.DataFrame({"Occurence": occurences, "Entry": entries,
                                "Stop_Loss": stop_losses, "Take_Profit": take_profits})
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"])
        return plot_df

    def DoubleTop(self, RR):
        sp = self._sp
        min_range = sp["min_range_ratio"]
        sl_fib = sp["sl_fib"]
        tol = sp["tolerance"]
        conditions = (
            self.new_df['low_shift_0'].notna() &
            self.new_df['low_shift_2'].notna() &
            self.new_df['low_shift_4'].notna() &
            self.new_df['high_shift_1'].notna() &
            self.new_df['high_shift_3'].notna() &
            (self.new_df['low_shift_0'] < self.new_df['high_shift_1']) &
            (self.new_df['low_shift_2'] < self.new_df['high_shift_1']) &
            (self.new_df['high_shift_3'] > self.new_df['low_shift_2']) &
            (self.new_df['low_shift_0'] < self.new_df['low_shift_2']) &
            (self.new_df['high_shift_3'] < self.new_df['high_shift_1']) &
            (self.new_df['low_shift_4'] < self.new_df['low_shift_2']) &
            ((self.new_df['high_shift_1'] - self.new_df['low_shift_0']) > (min_range * (self.new_df['high_shift_1'] - self.new_df['low_shift_2'])))
        )
        new_df_filtered = self.new_df[conditions]
        occurences, entries, stop_losses, take_profits = [], [], [], []
        if not new_df_filtered.empty:
            lower_tolerance = new_df_filtered['high_shift_3'] - (new_df_filtered['low_shift_2'] * np.tan(tol))
            upper_tolerance = new_df_filtered['high_shift_3'] + (new_df_filtered['low_shift_2'] * np.tan(tol))
            mask = (upper_tolerance >= new_df_filtered['high_shift_1']) & (new_df_filtered['high_shift_1'] >= lower_tolerance)
            valid_df = new_df_filtered[mask]
            if not valid_df.empty:
                occurences = valid_df.index.tolist()
                block_range = valid_df['high_shift_3'] - valid_df['low_shift_2']
                entry_price = valid_df['low_shift_2'] - block_range
                stop_loss = valid_df['high_shift_1'] + (block_range * sl_fib)
                take_profit = entry_price - ((stop_loss - entry_price) * RR)
                entries = entry_price.tolist()
                stop_losses = stop_loss.tolist()
                take_profits = take_profit.tolist()
        plot_df = pd.DataFrame({"Occurence": occurences, "Entry": entries,
                                "Stop_Loss": stop_losses, "Take_Profit": take_profits})
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"])
        return plot_df

    def TripleTop(self, RR):
        sp = self._sp
        min_range = sp["min_range_ratio"]
        sl_fib = sp["sl_fib"]
        conditions = (
            self.new_df['low_shift_0'].notna() &
            self.new_df['low_shift_2'].notna() &
            self.new_df['low_shift_4'].notna() &
            self.new_df['high_shift_1'].notna() &
            self.new_df['high_shift_3'].notna() &
            (self.new_df['low_shift_0'] < self.new_df['high_shift_1']) &
            (self.new_df['low_shift_2'] < self.new_df['high_shift_1']) &
            (self.new_df['high_shift_3'] > self.new_df['low_shift_2']) &
            (self.new_df['low_shift_0'] < self.new_df['low_shift_2']) &
            (self.new_df['high_shift_3'] < self.new_df['high_shift_1']) &
            (self.new_df['low_shift_4'] < self.new_df['low_shift_2']) &
            ((self.new_df['high_shift_1'] - self.new_df['low_shift_0']) > (min_range * (self.new_df['high_shift_1'] - self.new_df['low_shift_2'])))
        )
        new_df_filtered = self.new_df[conditions]
        occurences, entries, stop_losses, take_profits = [], [], [], []
        if not new_df_filtered.empty:
            occurences = new_df_filtered.index.tolist()
            block_range = new_df_filtered['high_shift_3'] - new_df_filtered['low_shift_2']
            entry_price = new_df_filtered['low_shift_2'] - block_range
            stop_loss = new_df_filtered['high_shift_1'] + (block_range * sl_fib)
            take_profit = entry_price - ((stop_loss - entry_price) * RR)
            entries = entry_price.tolist()
            stop_losses = stop_loss.tolist()
            take_profits = take_profit.tolist()
        plot_df = pd.DataFrame({"Occurence": occurences, "Entry": entries,
                                "Stop_Loss": stop_losses, "Take_Profit": take_profits})
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"])
        return plot_df

STRATEGIES = ["Noir", "BreakerBlock", "DoubleTop", "TripleTop"]
STRAT_IDX = {s: i for i, s in enumerate(STRATEGIES)}
STRATEGY_PARAM_KEYS = {s: list(STRATEGY_PARAM_BOUNDS[s].keys()) for s in STRATEGIES}
STRATEGY_PARAM_NDIMS = {s: len(STRATEGY_PARAM_KEYS[s]) for s in STRATEGIES}

POLY_MIN, POLY_MAX = 2, 14
WIN_MIN, WIN_MAX = 3, 19
ORDER_MIN, ORDER_MAX = 2, 14

DEFAULT_RR = 5.0
DEFAULT_RISK_AMOUNT = 25.0
DEFAULT_RISK_TYPE = "fixed"
DEFAULT_INITIAL_BALANCE = 1000.0
DEFAULT_MIN_TRADES = 10

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M10": mt5.TIMEFRAME_M10,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
}


def _fetch_data(symbol, timeframe_name, start, end):
    rates = mt5.copy_rates_range(symbol, TF_MAP[timeframe_name], start, end)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    rename_map = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tick_volume": "Volume", "real_volume": "Volume",
    }
    df.rename(columns=rename_map, inplace=True)
    df.set_index("time", inplace=True)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if "spread" in df.columns:
        cols.append("spread")
    ohlcv = df[cols].copy()
    if "spread" not in ohlcv.columns:
        ohlcv["spread"] = 0.0
    return ohlcv


def _clean_and_detect(df, polyorder, window_length, order):
    for col in ["smoothed_close", "Is_High", "Is_Low"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    close_prices = df["Close"].to_numpy()
    smoothed = savgol_filter(close_prices, window_length, polyorder)
    df["smoothed_close"] = smoothed
    highs = argrelextrema(smoothed, np.greater, mode="wrap", order=order)[0]
    lows = argrelextrema(smoothed, np.less, mode="wrap", order=order)[0]
    df.loc[df.index[highs], "Is_High"] = df["High"].iloc[highs]
    df.loc[df.index[lows], "Is_Low"] = df["Low"].iloc[lows]


def _run_backtest(df, plot_df, rr, initial_balance, risk_amount,
                  point_size, min_stop_dist, spread_arr):
    balance = initial_balance
    wins, losses, neither = 0, 0, 0
    results = []

    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    index_arr = df.index

    for trade in plot_df.itertuples():
        entry_price = float(trade.Entry)
        stop_loss = float(trade.Stop_Loss)
        take_profit = float(trade.Take_Profit)
        occurrence_time = trade.Occurence

        sl_dist = abs(entry_price - stop_loss)
        if sl_dist < min_stop_dist:
            continue

        try:
            occ_loc = index_arr.get_loc(occurrence_time)
        except KeyError:
            neither += 1
            continue

        future_high = high_arr[occ_loc + 1:]
        future_low = low_arr[occ_loc + 1:]
        future_spread = spread_arr[occ_loc + 1:]

        is_buy = stop_loss < entry_price

        if is_buy:
            entry_mask = (future_low + future_spread) <= entry_price
        else:
            entry_mask = future_high >= entry_price

        if not entry_mask.any():
            neither += 1
            results.append({
                "Occurrence": occurrence_time, "Entry": entry_price,
                "Stop_Loss": stop_loss, "Take_Profit": take_profit,
                "Result": "Pending", "Balance": balance,
            })
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
            balance += risk_amount * rr
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

    return pd.DataFrame(results), wins, losses, neither


def _base_result(polyorder, window_length, order, strategy_name, strategy_params=None):
    if strategy_params is None:
        strategy_params = {}
    return {
        "polyorder": polyorder, "window_length": window_length,
        "order": order, "strategy": strategy_name,
        "strategy_params": strategy_params.copy(),
        "wins": 0, "losses": 0, "neither": 0,
        "total_trades": 0, "final_balance": 0, "profit": 0,
        "win_rate": 0.0, "profit_factor": 0.0,
    }


def _evaluate(df, point_size, min_stop_dist, spread_arr,
              polyorder, window_length, order, strategy_name,
              rr, risk_amount, initial_balance, min_trades,
              strategy_params=None):
    if strategy_params is None:
        strategy_params = {}
    br = _base_result(polyorder, window_length, order, strategy_name, strategy_params)

    _clean_and_detect(df, polyorder, window_length, order)

    strategy = ParameterizedStrategy(df)
    strategy.configure(strategy_name, strategy_params)
    plot_df = getattr(strategy, strategy_name)(RR=rr)
    if plot_df.empty:
        return -1.0, {**br, "error": "no trade signals"}

    backtest_df, wins, losses, neither = _run_backtest(
        df, plot_df, rr, initial_balance, risk_amount,
        point_size=point_size, min_stop_dist=min_stop_dist, spread_arr=spread_arr,
    )
    if backtest_df.empty:
        return -1.0, {**br, "wins": wins, "losses": losses, "error": "backtest empty"}

    total_trades = wins + losses
    if total_trades < min_trades:
        return -1.5, {**br, "score": -1.5, "wins": wins, "losses": losses,
                       "total_trades": total_trades,
                       "error": f"only {total_trades} trades (<{min_trades})"}

    final_balance = backtest_df.iloc[-1]["Balance"]
    profit = final_balance - initial_balance
    win_rate = wins / total_trades
    profit_factor = (wins * rr) / losses if losses > 0 else float("inf")
    score = profit
    details = {
        **br, "score": score,
        "wins": wins, "losses": losses, "neither": neither,
        "total_trades": total_trades,
        "final_balance": round(final_balance, 2),
        "profit": round(profit, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
    }
    return score, details


def _strat_ndims(strategies):
    if len(strategies) == 1:
        return 3 + STRATEGY_PARAM_NDIMS[strategies[0]]
    return 4


def _strat_param_keys(strategies):
    if len(strategies) == 1:
        return STRATEGY_PARAM_KEYS[strategies[0]]
    return []


def _param_defaults_from_norm(strategies, x_norm):
    if len(strategies) == 1:
        s = strategies[0]
        keys = STRATEGY_PARAM_KEYS[s]
        bounds = STRATEGY_PARAM_BOUNDS[s]
        return {k: x_norm[3 + i] * (bounds[k][1] - bounds[k][0]) + bounds[k][0]
                for i, k in enumerate(keys)}
    return {}


def _encode_params(polyorder, window_length, order, strategy_name_or_params,
                   strategies=None, strategy_params=None):
    if strategies is None:
        strategies = STRATEGIES
    dims = _strat_ndims(strategies)
    arr = np.zeros(dims)
    arr[0] = (polyorder - POLY_MIN) / (POLY_MAX - POLY_MIN)
    arr[1] = (window_length - WIN_MIN) / (WIN_MAX - WIN_MIN)
    arr[2] = (order - ORDER_MIN) / (ORDER_MAX - ORDER_MIN)
    if len(strategies) == 1:
        s = strategies[0]
        if strategy_params is None:
            strategy_params = {}
        bounds = STRATEGY_PARAM_BOUNDS[s]
        keys = STRATEGY_PARAM_KEYS[s]
        for i, k in enumerate(keys):
            val = strategy_params.get(k, STRATEGY_PARAM_DEFAULTS[s][k])
            arr[3 + i] = (val - bounds[k][0]) / (bounds[k][1] - bounds[k][0])
    else:
        n_strat = len(strategies)
        idx = strategies.index(strategy_name_or_params) if strategy_name_or_params in strategies else 0
        arr[3] = idx / (n_strat - 1) if n_strat > 1 else 0.5
    return arr


def _decode_params(x_norm, strategies=None):
    if strategies is None:
        strategies = STRATEGIES
    polyorder = int(round(x_norm[0] * (POLY_MAX - POLY_MIN) + POLY_MIN))
    window_length = int(round(x_norm[1] * (WIN_MAX - WIN_MIN) + WIN_MIN))
    order = int(round(x_norm[2] * (ORDER_MAX - ORDER_MIN) + ORDER_MIN))
    if len(strategies) == 1:
        s = strategies[0]
        strategy_params = _param_defaults_from_norm(strategies, x_norm)
        return polyorder, window_length, order, s, strategy_params
    n_strat = len(strategies)
    strat_idx = int(round(x_norm[3] * (n_strat - 1))) if n_strat > 1 else 0
    strategy_name = strategies[min(strat_idx, n_strat - 1)]
    return polyorder, window_length, order, strategy_name, {}


def _sobol_samples(n, strategies=None, seed=42):
    dims = _strat_ndims(strategies)
    pow2 = 2 ** int(np.floor(np.log2(n)))
    sampler = qmc.Sobol(d=dims, seed=seed)
    all_samples = []
    if pow2 >= 2:
        all_samples.extend(sampler.random(pow2))
    remaining = n - len(all_samples)
    if remaining > 0:
        rng = np.random.RandomState(seed + 1)
        all_samples.extend(rng.uniform(size=(remaining, dims)))
    return [_decode_params(s, strategies) for s in all_samples]


def _random_candidates(n, rng, strategies=None):
    if strategies is None:
        strategies = STRATEGIES
    candidates = []
    keys = _strat_param_keys(strategies)
    bounds_dict = STRATEGY_PARAM_BOUNDS.get(strategies[0], {}) if len(strategies) == 1 else {}
    for _ in range(n):
        polyorder = rng.randint(POLY_MIN, POLY_MAX + 1)
        window_length = rng.randint(WIN_MIN, WIN_MAX + 1)
        order = rng.randint(ORDER_MIN, ORDER_MAX + 1)
        window_length = max(window_length, polyorder + 2)
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, WIN_MAX)
        strategy_params = {}
        for k in keys:
            lo, hi = bounds_dict[k]
            strategy_params[k] = rng.uniform(lo, hi)
        if len(strategies) == 1:
            candidates.append((polyorder, window_length, order, strategies[0], strategy_params))
        else:
            strategy_name = rng.choice(strategies)
            candidates.append((polyorder, window_length, order, strategy_name, strategy_params))
    return candidates


def optimize(symbol, timeframe="M10", start_str=None, end_str=None,
             n_iterations=80, n_initial=20, rr=DEFAULT_RR,
             risk_amount=DEFAULT_RISK_AMOUNT, risk_type=DEFAULT_RISK_TYPE,
             initial_balance=DEFAULT_INITIAL_BALANCE, min_trades=DEFAULT_MIN_TRADES,
             strategies=None, progress_callback=None):
    if not mt5.initialize():
        msg = "[ERROR] MT5 initialization failed."
        if progress_callback:
            progress_callback(msg)
        return None, []

    try:
        if start_str is None:
            start = datetime.now() - timedelta(days=180)
        else:
            start = datetime.strptime(start_str, "%Y-%m-%d")
        if end_str is None:
            end = datetime.now()
        else:
            end = datetime.strptime(end_str, "%Y-%m-%d")

        if strategies is None:
            strategies = STRATEGIES
        strat_list = list(strategies)

        def _report(msg):
            if progress_callback:
                progress_callback(msg)

        _report(f"[OPTIMIZE] {symbol} {timeframe} | {start.date()} → {end.date()}")
        _report(f"[OPTIMIZE] Strategies: {', '.join(strat_list)}")
        _report(f"[OPTIMIZE] Search space: polyorder=[{POLY_MIN},{POLY_MAX}] "
                f"window_length=[{WIN_MIN},{WIN_MAX}] order=[{ORDER_MIN},{ORDER_MAX}]")
        _report("")

        _report("[FETCH] Loading historical data...")
        raw_df = _fetch_data(symbol, timeframe, start, end)
        if raw_df is None or len(raw_df) < 50:
            _report(f"[ERROR] Insufficient data for {symbol} ({len(raw_df) if raw_df is not None else 0} rows)")
            return None, []

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            _report(f"[ERROR] Cannot retrieve symbol info for {symbol}")
            return None, []
        point_size = symbol_info.point
        min_stop_dist = symbol_info.trade_stops_level * point_size
        raw_spreads = raw_df["spread"].to_numpy() if "spread" in raw_df.columns else np.zeros(len(raw_df))
        spread_arr = raw_spreads * point_size

        _report(f"[FETCH] Loaded {len(raw_df)} candles for {symbol}, "
                f"point_size={point_size}, min_stop_dist={min_stop_dist:.5f}")
        _report(f"[OPTIMIZE] Initial {n_initial} Sobol samples + {n_iterations - n_initial} Bayesian iterations")
        _report("")

        scores = []
        X = []
        history = []

        n_init = min(n_initial, n_iterations)
        initial_params = _sobol_samples(n_init, strat_list)
        rng = np.random.RandomState(42)

        for step in range(n_iterations):
            if step < n_init:
                polyorder, window_length, order, strategy_name, strategy_params = initial_params[step]
            else:
                X_arr = np.array(X)
                y_arr = np.array(scores)
                y_std = y_arr.std()
                if y_std < 1e-6:
                    y_std = 1.0

                kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                          * Matern(length_scale=1.0, nu=2.5)
                          + WhiteKernel(noise_level=0.1 * y_std, noise_level_bounds=(1e-6, 1.0)))
                surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                                      random_state=42, normalize_y=True)
                surrogate.fit(X_arr, y_arr)

                candidates = _random_candidates(500, rng, strat_list)
                X_candidates = np.array([_encode_params(*c[:3], c[3], strat_list,
                                        strategy_params=c[4] if len(c) > 4 else None)
                                        for c in candidates])
                preds, stds = surrogate.predict(X_candidates, return_std=True)
                y_best = y_arr.max()

                with np.errstate(divide="ignore", invalid="ignore"):
                    imp = preds - y_best
                    Z = imp / stds
                    ei = imp * _norm_cdf(Z) + stds * _norm_pdf(Z)
                    ei[stds <= 1e-8] = 0.0

                best_idx = np.argmax(ei)
                if ei.max() <= 0:
                    best_idx = np.argmax(stds)
                polyorder, window_length, order, strategy_name, strategy_params = candidates[best_idx]

            polyorder = np.clip(polyorder, POLY_MIN, POLY_MAX)
            order = np.clip(order, ORDER_MIN, ORDER_MAX)
            window_length = np.clip(window_length, WIN_MIN, WIN_MAX)
            window_length = max(window_length, polyorder + 2)
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, WIN_MAX)

            df = raw_df.copy()
            score, details = _evaluate(
                df, point_size, min_stop_dist, spread_arr,
                int(polyorder), int(window_length), int(order), strategy_name,
                rr, risk_amount, initial_balance, min_trades,
                strategy_params=strategy_params,
            )

            X.append(_encode_params(int(polyorder), int(window_length), int(order), strategy_name, strat_list,
                                     strategy_params=strategy_params))
            scores.append(score)
            history.append(details)

            sp_str = ""
            if strategy_params:
                sp_str = " " + " ".join(f"{k}={v:.3f}" for k, v in sorted(strategy_params.items()))
            status = "✓" if score > 0 else "✗"
            _report(f"[{status} {step+1}/{n_iterations}] P={polyorder} W={window_length} "
                    f"O={order} S={strategy_name:<14}{sp_str} → "
                    f"profit={details.get('profit', 0):>8}  "
                    f"trades={details.get('total_trades', 0):>3}  "
                    f"wr={details.get('win_rate', 0):>6}  "
                    f"score={score:.2f}")

        best_idx = int(np.argmax(scores))
        best = history[best_idx]

        _report("")
        _report("═══════════════════════════════════════════")
        _report(f" BEST: P={best['polyorder']} W={best['window_length']} "
                f"O={best['order']} S={best['strategy']}")
        sp = best.get("strategy_params", {})
        if sp:
            _report(f"       strategy params: {' '.join(f'{k}={v:.3f}' for k,v in sorted(sp.items()))}")
        _report(f"       profit={best['profit']}  trades={best['total_trades']}  "
                f"wr={best['win_rate']}  pf={best['profit_factor']}")
        _report("═══════════════════════════════════════════")

        settings = load_settings()
        if symbol not in settings:
            settings[symbol] = {}
        if isinstance(settings[symbol], dict):
            settings[symbol]["polyorder"] = int(best["polyorder"])
            settings[symbol]["window_length"] = int(best["window_length"])
            settings[symbol]["order"] = int(best["order"])
            settings[symbol]["best_strategy"] = best["strategy"]
            if sp:
                settings[symbol]["strategy_params"] = {k: round(v, 4) for k, v in sp.items()}
            _report(f"[DONE] Best strategy: {best['strategy']}")
        save_settings(settings)

        save_opt_history(symbol, timeframe, strat_list, n_iterations,
                         start_str if start_str else str(start.date()),
                         end_str if end_str else str(end.date()),
                         best, history)
        _report(f"[DONE] Optimization saved to history.")

        _report(f"[DONE] Best params saved to settings.json for {symbol}")

        return best, history

    finally:
        mt5.shutdown()


def save_opt_history(symbol, timeframe, strategies, n_iterations,
                     start_date, end_date, best, history):
    os.makedirs(os.path.dirname(OPT_HISTORY_PATH), exist_ok=True)
    records = load_opt_history()
    entry = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategies": strategies,
        "n_iterations": n_iterations,
        "start_date": start_date,
        "end_date": end_date,
        "best": best,
    }
    records.insert(0, entry)
    max_records = 200
    if len(records) > max_records:
        records = records[:max_records]
    with open(OPT_HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def load_opt_history():
    if os.path.exists(OPT_HISTORY_PATH):
        try:
            with open(OPT_HISTORY_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def delete_opt_history(record_id):
    records = load_opt_history()
    records = [r for r in records if r.get("id") != record_id]
    with open(OPT_HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)
    return records


def _norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x):
    from scipy.special import erf
    return erf(x)


def _norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
