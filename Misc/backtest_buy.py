import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
from datetime import datetime, timedelta
from strategy_buy import StrategyBuy
import trendet

# Define the forex pairs
symbols = ["USDJPY", "EURJPY", "AUDJPY", "GBPJPY"]

# Define the timeframes you want to test
timeframes = {
    "M1": mt5.TIMEFRAME_M1,
}

# Define the start and end times for the data
start_time = pd.to_datetime("2024-07-01 00:00:00")
end_time = pd.to_datetime("2024-07-05 23:00:00")

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
    filename = f"{symbol}_data_{timeframe_name}.csv"
    df.to_csv(filename, index=False)
    mt5.shutdown()

def prepData(symbol, timeframe_name, visualize=False):
    filename = f"{symbol}_data_{timeframe_name}.csv"
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

def cleanData(df, visualize=False):
    window_length = 15
    polyorder = 8
    smoothed_close = savgol_filter(df["Close"], window_length, polyorder)
    df["smoothed_close"] = smoothed_close

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["Close"], label="Close Price")
        plt.plot(df.index, df["smoothed_close"], label="Smoothed Close Price")
        plt.legend()
        plt.show()

def detectPivotPoints(df, visualize=False):
    order = 5
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

def identify_trends(df, start_time, end_time):
    trends_df = trendet.identify_all_trends(
        df,
        country="global",
        from_date=start_time,
        to_date=end_time,
        window_size=5,
        identify='both'
    )
    
    trend_map = {'Up Trend': 1, 'Down Trend': -1, 'No Trend': 0}
    trends_df['Trend'] = trends_df['Up Trend'].map(trend_map).fillna(trends_df['Down Trend'].map(trend_map))
    df['Trend'] = trends_df['Trend']
    return df

def backtest(df, plot_df, RR, balance, risk_amount, risk_type):
    results = []
    wins = 0
    losses = 0
    neither = 0
    running = 0

    for trade in plot_df.itertuples():
        entry_price = trade.Entry
        stop_loss = trade.Stop_Loss
        take_profit = trade.Take_Profit
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

            if high_price >= entry_price:
                entry_reached = True

            if entry_reached:
                if low_price <= stop_loss:
                    price_reached_stop_loss = True
                    balance = (balance - Risk)
                    break

                if high_price >= take_profit:
                    price_reached_take_profit = True
                    balance = balance + (Risk * RR)
                    break

        if price_reached_stop_loss:
            result = "SL"
            losses += 1
        elif price_reached_take_profit:
            result = "TP"
            wins += 1
        else:
            result = "Neither Hit"
            neither += 1

        trade_result = {
            "Occurrence": occurrence_time,
            "Entry": entry_price,
            "Stop_Loss": stop_loss,
            "Take_Profit": take_profit,
            "Result": result,
            "Balance": balance,
        }

        results.append(trade_result)

    backtest_df = pd.DataFrame(results)
    print(backtest_df)
    return backtest_df, wins, losses, neither

def plot_balance_graph(backtest_results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(backtest_results_df["Occurrence"], backtest_results_df["Balance"], marker="")
    plt.title("ACCOUNT GROWTH")
    plt.xlabel("Trade Time")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.show()

summary_results = []
strategies = ["SMSstrategy"]

for symbol in symbols:
    for timeframe_name, timeframe in timeframes.items():
        print(f"Running backtest for {symbol} on {timeframe_name} timeframe...")
        get_historical_data(symbol, timeframe, timeframe_name, start_time, end_time)
        df = prepData(symbol, timeframe_name, visualize=False)
        cleanData(df)
        detectPivotPoints(df)
        # df = identify_trends(df, start_time, end_time)
        for strategy_name in strategies:
            strategy = StrategyBuy(df)
            plot_df = getattr(strategy, strategy_name)(RR=5)

            balance = 100
            risk_amount = 10
            risk_type = ""
            backtest_results_df, wins, losses, neither = backtest(df, plot_df, RR=5, balance=balance, risk_amount=risk_amount, risk_type=risk_type)

            final_balance = 1

            if wins != 0:
                win_rate = (wins/(wins+losses)) * 100
            else:
                win_rate = 0

            current_count = 0
            highest_count = 0

            # Iterate through the filtered DataFrame to count consecutive stop loss hits
            for i in range(len(backtest_results_df)):
                if (
                    backtest_results_df.iloc[i]["Result"]
                    == backtest_results_df.iloc[i - 1]["Result"]
                    == "SL"
                ):
                    # If the current row is the first row or the occurrence time is one day after the previous row
                    current_count += 1
                    if current_count > highest_count:
                        # If the current count is higher than the highest count, update the highest count
                        highest_count = current_count

                else:
                    # If the occurrence time is not consecutive, reset the current count
                    current_count = 1

            # Append the summary results for the current timeframe to the list


            summary_results.append({
                "Symbol": symbol,
                "Timeframe": timeframe_name,
                "Strategy": strategy_name,
                "Wins": wins,
                "Losses": losses,
                "Neither": neither,
                "Consecutive SL": highest_count,
                "Final Balance": final_balance,
                "Win Rate": win_rate
            })


def plot(trade):
    specific_datetime = trade.Occurence
    dfpl = df[specific_datetime - timedelta(days=7):specific_datetime + timedelta(days=7)]
    apd = [
        mpf.make_addplot(dfpl["Is_High"], scatter=True, markersize=30, marker="x", color="b"),
        mpf.make_addplot(dfpl["Is_Low"], scatter=True, markersize=30, marker="x", color="r"),
    ]
    mpf.plot(
        dfpl,
        type="candle",
        style="nightclouds",
        title=f"{symbol}",
        warn_too_much_data=9999999999,
        addplot=apd,
        vlines=[specific_datetime],
        hlines=dict(hlines=[trade.Stop_Loss, trade.Take_Profit, trade.Entry], colors=['r','g','b'], linestyle='-'),
    )

 # Uncomment this to plot charts

summary_df = pd.DataFrame(summary_results)
print(summary_df)
summary_df.to_csv("backtest_summary.csv", index=False)
plot_balance_graph(backtest_results_df)

for trade in plot_df.itertuples():
    plot(trade) 