import pandas as pd
import pandas_ta as ta
import math

class Strategy:
    def __init__(self, dataframe):
        self.occurences = []
        self.entries = []
        self.stop_losses = []
        self.take_profits = []
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

        self.new_df_filtered = self.new_df[conditions]

        if not self.new_df_filtered.empty:
            self.occurences.extend(self.new_df_filtered.index.tolist())
            block_range = self.new_df_filtered['high_shift_3'] - self.new_df_filtered['low_shift_4']
            entry_price = self.new_df_filtered['high_shift_5']
            stop_loss = self.new_df_filtered['high_shift_5'] + (block_range * 1.1)
            take_profit = entry_price - ((stop_loss - entry_price) * RR)

            self.entries.extend(entry_price.tolist())
            self.stop_losses.extend(stop_loss.tolist())
            self.take_profits.extend(take_profit.tolist())

        plot_df = pd.DataFrame({
            "Occurence": self.occurences,
            "Entry": self.entries,
            "Stop_Loss": self.stop_losses,
            "Take_Profit": self.take_profits,
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

        self.new_df_filtered = self.new_df[conditions]

        if not self.new_df_filtered.empty:
            self.occurences.extend(self.new_df_filtered.index.tolist())
            block_range = self.new_df_filtered['high_shift_3'] - self.new_df_filtered['low_shift_2']
            entry_price = self.new_df_filtered['low_shift_2']
            stop_loss = self.new_df_filtered['high_shift_1'] + (block_range * self.fibonacci_levels[1])
            take_profit = entry_price - ((stop_loss - entry_price) * RR)

            self.entries.extend(entry_price.tolist())
            self.stop_losses.extend(stop_loss.tolist())
            self.take_profits.extend(take_profit.tolist())

        plot_df = pd.DataFrame({
            "Occurence": self.occurences,
            "Entry": self.entries,
            "Stop_Loss": self.stop_losses,
            "Take_Profit": self.take_profits,
        })
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df

    def DoubleTop(self, RR):
        tolerance = 0.05
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

        self.new_df_filtered = self.new_df[conditions]

        if not self.new_df_filtered.empty:
            lower_tolerance = self.new_df_filtered['high_shift_3'] - (self.new_df_filtered['low_shift_2'] * math.tan(0.02))
            upper_tolerance = self.new_df_filtered['high_shift_3'] + (self.new_df_filtered['low_shift_2'] * math.tan(0.02))
            mask = (upper_tolerance >= self.new_df_filtered['high_shift_1']) & (self.new_df_filtered['high_shift_1'] >= lower_tolerance)

            valid_df = self.new_df_filtered[mask]
            if not valid_df.empty:
                self.occurences.extend(valid_df.index.tolist())
                block_range = valid_df['high_shift_3'] - valid_df['low_shift_2']
                entry_price = valid_df['low_shift_2'] - block_range
                stop_loss = valid_df['high_shift_1'] + (block_range * self.fibonacci_levels[1])
                take_profit = entry_price - ((stop_loss - entry_price) * RR)

                self.entries.extend(entry_price.tolist())
                self.stop_losses.extend(stop_loss.tolist())
                self.take_profits.extend(take_profit.tolist())

        plot_df = pd.DataFrame({
            "Occurence": self.occurences,
            "Entry": self.entries,
            "Stop_Loss": self.stop_losses,
            "Take_Profit": self.take_profits,
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

        self.new_df_filtered = self.new_df[conditions]

        if not self.new_df_filtered.empty:
            self.occurences.extend(self.new_df_filtered.index.tolist())
            block_range = self.new_df_filtered['high_shift_3'] - self.new_df_filtered['low_shift_2']
            entry_price = self.new_df_filtered['low_shift_2'] - block_range
            stop_loss = self.new_df_filtered['high_shift_1'] + (block_range * self.fibonacci_levels[1])
            take_profit = entry_price - ((stop_loss - entry_price) * RR)

            self.entries.extend(entry_price.tolist())
            self.stop_losses.extend(stop_loss.tolist())
            self.take_profits.extend(take_profit.tolist())

        plot_df = pd.DataFrame({
            "Occurence": self.occurences,
            "Entry": self.entries,
            "Stop_Loss": self.stop_losses,
            "Take_Profit": self.take_profits,
        })
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df
