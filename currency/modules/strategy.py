import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, dataframe, symbol=None):
        self.symbol = symbol
        self.fibonacci_levels = [0, 0.272, 0.382, 0.5, 0.618, 0.786, 1, 1.361, 1.836]
        self.new_df = dataframe.loc[dataframe["Is_High"].notna() | dataframe["Is_Low"].notna()].copy()

        # Prepare shifted columns
        self.new_df['low_shift_0'] = self.new_df['Is_Low'].shift(0)
        self.new_df['low_shift_2'] = self.new_df['Is_Low'].shift(2)
        self.new_df['low_shift_4'] = self.new_df['Is_Low'].shift(4)
        self.new_df['high_shift_1'] = self.new_df['Is_High'].shift(1)
        self.new_df['high_shift_3'] = self.new_df['Is_High'].shift(3)
        self.new_df['high_shift_5'] = self.new_df['Is_High'].shift(5)

    def Noir(self, RR):
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
            ((self.new_df['low_shift_2'] - self.new_df['low_shift_0']) > (110 / 100 * (self.new_df['high_shift_1'] - self.new_df['low_shift_2']))) &
            ((self.new_df['low_shift_2'] - self.new_df['low_shift_0']) < (300 / 100 * (self.new_df['high_shift_1'] - self.new_df['low_shift_2']))) &
            ((self.new_df['high_shift_3'] - self.new_df['low_shift_2']) > (120 / 100 * (self.new_df['high_shift_3'] - self.new_df['low_shift_4']))) &
            ((self.new_df['high_shift_3'] - self.new_df['low_shift_2']) < (300 / 100 * (self.new_df['high_shift_3'] - self.new_df['low_shift_4'])))
        )

        new_df_filtered = self.new_df[conditions]

        occurences = []
        entries = []
        stop_losses = []
        take_profits = []

        if not new_df_filtered.empty:
            occurences = new_df_filtered.index.tolist()
            block_range = new_df_filtered['high_shift_3'] - new_df_filtered['low_shift_4']
            entry_price = new_df_filtered['high_shift_5']
            stop_loss = new_df_filtered['high_shift_5'] + (block_range * 1.1)
            take_profit = entry_price - ((stop_loss - entry_price) * RR)

            entries = entry_price.tolist()
            stop_losses = stop_loss.tolist()
            take_profits = take_profit.tolist()

        plot_df = pd.DataFrame({
            "Occurence": occurences,
            "Entry": entries,
            "Stop_Loss": stop_losses,
            "Take_Profit": take_profits,
        })
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df

    def BreakerBlock(self, RR):
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
            ((self.new_df['high_shift_1'] - self.new_df['low_shift_0']) > (200 / 100 * (self.new_df['high_shift_1'] - self.new_df['low_shift_2'])))
        )

        new_df_filtered = self.new_df[conditions]

        occurences = []
        entries = []
        stop_losses = []
        take_profits = []

        if not new_df_filtered.empty:
            occurences = new_df_filtered.index.tolist()
            block_range = new_df_filtered['high_shift_3'] - new_df_filtered['low_shift_2']
            entry_price = new_df_filtered['low_shift_2']
            stop_loss = new_df_filtered['high_shift_1'] + (block_range * self.fibonacci_levels[1])
            take_profit = entry_price - ((stop_loss - entry_price) * RR)

            entries = entry_price.tolist()
            stop_losses = stop_loss.tolist()
            take_profits = take_profit.tolist()

        plot_df = pd.DataFrame({
            "Occurence": occurences,
            "Entry": entries,
            "Stop_Loss": stop_losses,
            "Take_Profit": take_profits,
        })
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df

    def DoubleTop(self, RR):
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
            ((self.new_df['high_shift_1'] - self.new_df['low_shift_0']) > (200 / 100 * (self.new_df['high_shift_1'] - self.new_df['low_shift_2'])))
        )

        new_df_filtered = self.new_df[conditions]

        occurences = []
        entries = []
        stop_losses = []
        take_profits = []

        if not new_df_filtered.empty:
            lower_tolerance = new_df_filtered['high_shift_3'] - (new_df_filtered['low_shift_2'] * np.tan(0.02))
            upper_tolerance = new_df_filtered['high_shift_3'] + (new_df_filtered['low_shift_2'] * np.tan(0.02))
            mask = (upper_tolerance >= new_df_filtered['high_shift_1']) & (new_df_filtered['high_shift_1'] >= lower_tolerance)

            valid_df = new_df_filtered[mask]
            if not valid_df.empty:
                occurences = valid_df.index.tolist()
                block_range = valid_df['high_shift_3'] - valid_df['low_shift_2']
                entry_price = valid_df['low_shift_2'] - block_range
                stop_loss = valid_df['high_shift_1'] + (block_range * self.fibonacci_levels[1])
                take_profit = entry_price - ((stop_loss - entry_price) * RR)

                entries = entry_price.tolist()
                stop_losses = stop_loss.tolist()
                take_profits = take_profit.tolist()

        plot_df = pd.DataFrame({
            "Occurence": occurences,
            "Entry": entries,
            "Stop_Loss": stop_losses,
            "Take_Profit": take_profits,
        })
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df

    def TripleTop(self, RR):
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
            ((self.new_df['high_shift_1'] - self.new_df['low_shift_0']) > (200 / 100 * (self.new_df['high_shift_1'] - self.new_df['low_shift_2'])))
        )

        new_df_filtered = self.new_df[conditions]

        occurences = []
        entries = []
        stop_losses = []
        take_profits = []

        if not new_df_filtered.empty:
            occurences = new_df_filtered.index.tolist()
            block_range = new_df_filtered['high_shift_3'] - new_df_filtered['low_shift_2']
            entry_price = new_df_filtered['low_shift_2'] - block_range
            stop_loss = new_df_filtered['high_shift_1'] + (block_range * self.fibonacci_levels[1])
            take_profit = entry_price - ((stop_loss - entry_price) * RR)

            entries = entry_price.tolist()
            stop_losses = stop_loss.tolist()
            take_profits = take_profit.tolist()

        plot_df = pd.DataFrame({
            "Occurence": occurences,
            "Entry": entries,
            "Stop_Loss": stop_losses,
            "Take_Profit": take_profits,
        })
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df

    def MLPattern(self, RR):
        from .ml_pattern import get_all_pivot_sequences, predict_pattern_probability
        
        sequences, trigger_indices, is_buy_list = get_all_pivot_sequences(self.new_df)
        print(f"[INFO] Evaluating {len(sequences)} swing patterns using ML model for {symbol}...")
        
        occurences = []
        entries = []
        stop_losses = []
        take_profits = []

        symbol = self.symbol if self.symbol else "EURUSD"

        for seq, trig_time, is_buy in zip(sequences, trigger_indices, is_buy_list):
            prob = predict_pattern_probability(symbol, seq)
            if prob >= 0.58:  # Trigger on high probability predictions
                entry_price = seq[0]["val"]
                wave_size = abs(seq[1]["val"] - seq[2]["val"])
                if wave_size == 0:
                    wave_size = 1e-4

                if is_buy:
                    stop_loss = entry_price - (wave_size * 0.5)
                    take_profit = entry_price + (entry_price - stop_loss) * RR
                else:
                    stop_loss = entry_price + (wave_size * 0.5)
                    take_profit = entry_price - (stop_loss - entry_price) * RR

                occurences.append(trig_time)
                entries.append(entry_price)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

        plot_df = pd.DataFrame({
            "Occurence": occurences,
            "Entry": entries,
            "Stop_Loss": stop_losses,
            "Take_Profit": take_profits,
        })
        if not plot_df.empty:
            plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
                plot_df["Entry"] - plot_df["Stop_Loss"]
            )
        else:
            plot_df["Risk_to_Reward_Ratio"] = pd.Series(dtype='float64')

        return plot_df
