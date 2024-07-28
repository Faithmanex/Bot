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

        for idx, row in self.new_df_filtered.iterrows():
            self.occurences.append(idx)
            block_range = row['high_shift_3'] - row['low_shift_4']
            entry_price = row['high_shift_5']
            self.entries.append(entry_price)
            stop_loss = row['high_shift_5'] + (block_range * 1.1)
            take_profit = entry_price - ((stop_loss - entry_price) * RR)
            self.stop_losses.append(stop_loss)
            self.take_profits.append(take_profit)

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

        for idx, row in self.new_df_filtered.iterrows():
            self.occurences.append(idx)
            block_range = row['high_shift_3'] - row['low_shift_2']
            entry_price = row['low_shift_2']
            self.entries.append(entry_price)
            stop_loss = row['high_shift_1'] + (block_range * self.fibonacci_levels[1])
            take_profit = entry_price - ((stop_loss - entry_price) * RR)
            self.stop_losses.append(stop_loss)
            self.take_profits.append(take_profit)

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

        for idx, row in self.new_df_filtered.iterrows():
            lower_tolerance = row['high_shift_3'] - (row['low_shift_2'] * math.tan(0.02))
            upper_tolerance = row['high_shift_3'] + (row['low_shift_2'] * math.tan(0.02))
            if upper_tolerance >= row['high_shift_1'] >= lower_tolerance:
                self.occurences.append(idx)
                block_range = row['high_shift_3'] - row['low_shift_2']
                entry_price = row['low_shift_2'] - block_range
                self.entries.append(entry_price)
                stop_loss = row['high_shift_1'] + (block_range * self.fibonacci_levels[1])
                take_profit = entry_price - ((stop_loss - entry_price) * RR)
                self.stop_losses.append(stop_loss)
                self.take_profits.append(take_profit)

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

        for idx, row in self.new_df_filtered.iterrows():
            self.occurences.append(idx)
            block_range = row['high_shift_3'] - row['low_shift_2']
            entry_price = row['low_shift_2'] - block_range
            self.entries.append(entry_price)
            stop_loss = row['high_shift_1'] + (block_range * self.fibonacci_levels[1])
            take_profit = entry_price - ((stop_loss - entry_price) * RR)
            self.stop_losses.append(stop_loss)
            self.take_profits.append(take_profit)

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
