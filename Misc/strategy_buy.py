import pandas as pd
import pandas_ta as ta
import math

class StrategyBuy:
    def __init__(self, dataframe):
        self.occurences = []
        self.entries = []
        self.stop_losses = []
        self.take_profits = []
        self.fibonacci_levels = [0, 0.272, 0.382, 0.5, 0.618, 0.786, 1, 1.361, 1.836]
        self.new_df = dataframe.loc[dataframe["Is_High"].notna() | dataframe["Is_Low"].notna()]

    def breaker_block(self, RR):
        # Breaker Block
        for x in range(4, self.new_df.shape[0]):
            if (
                pd.notna(self.new_df.iloc[x]["Is_Low"]) and
                pd.notna(self.new_df.iloc[x - 1]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 2]["Is_Low"]) and
                pd.notna(self.new_df.iloc[x - 3]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 4]["Is_Low"])
            ):
                if (
                    (self.new_df.iloc[x]["Is_Low"] < self.new_df.iloc[x - 1]["Is_High"]) and
                    (self.new_df.iloc[x - 2]["Is_Low"] < self.new_df.iloc[x - 1]["Is_High"]) and
                    (self.new_df.iloc[x - 3]["Is_High"] > self.new_df.iloc[x - 2]["Is_Low"]) and
                    (self.new_df.iloc[x]["Is_Low"] < self.new_df.iloc[x - 2]["Is_Low"]) and
                    (self.new_df.iloc[x - 3]["Is_High"] < self.new_df.iloc[x - 1]["Is_High"]) and
                    (self.new_df.iloc[x - 4]["Is_Low"] < self.new_df.iloc[x - 2]["Is_Low"])
                ) and (
                    (self.new_df.iloc[x - 1]["Is_High"] - self.new_df.iloc[x]["Is_Low"]) >
                    (2 * (self.new_df.iloc[x - 1]["Is_High"] - self.new_df.iloc[x - 2]["Is_Low"]))
                ):
                    self.occurences.append(self.new_df.index[x])
                    block_range = self.new_df.iloc[x - 3]["Is_High"] - self.new_df.iloc[x - 2]["Is_Low"]

                    entry_price = self.new_df.iloc[x - 2]["Is_Low"]
                    self.entries.append(entry_price)

                    stop_loss = self.new_df.iloc[x - 1]["Is_High"] + (block_range * self.fibonacci_levels[1])
                    self.stop_losses.append(stop_loss)

                    take_profit = entry_price - ((stop_loss - entry_price) * RR)
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

    def AMSstrategy(self, RR):
        for x in range(5, self.new_df.shape[0]):
            if (
                pd.notna(self.new_df.iloc[x]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 1]["Is_Low"]) and
                pd.notna(self.new_df.iloc[x - 2]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 3]["Is_Low"]) and
                pd.notna(self.new_df.iloc[x - 4]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 5]["Is_Low"])
            ):
                if (
                    (self.new_df.iloc[x]["Is_High"] > self.new_df.iloc[x - 2]["Is_High"]) and
                    (self.new_df.iloc[x - 3]["Is_Low"] < self.new_df.iloc[x - 1]["Is_Low"]) and
                    (self.new_df.iloc[x - 1]["Is_Low"] < self.new_df.iloc[x - 2]["Is_High"]) and
                    (self.new_df.iloc[x - 4]["Is_High"] > self.new_df.iloc[x - 2]["Is_High"]) and
                    (self.new_df.iloc[x - 1]["Is_Low"] > self.new_df.iloc[x - 3]["Is_Low"])
                ) and (
                    (self.new_df.iloc[x]["Is_High"] - self.new_df.iloc[x - 2]["Is_High"]) >
                    (3 * (self.new_df.iloc[x - 1]["Is_Low"] - self.new_df.iloc[x - 2]["Is_High"]))
                ) and (
                    (self.new_df.iloc[x]["Is_High"] - self.new_df.iloc[x - 2]["Is_High"]) >
                    (1.2 * (self.new_df.iloc[x - 1]["Is_Low"] - self.new_df.iloc[x - 2]["Is_High"]))
                ):
                    self.occurences.append(self.new_df.index[x])
                    block_range = self.new_df.iloc[x - 3]["Is_Low"] - self.new_df.iloc[x - 4]["Is_High"]

                    entry_price = self.new_df.iloc[x - 3]["Is_Low"]
                    self.entries.append(entry_price)

                    stop_loss = self.new_df.iloc[x - 5]["Is_Low"] + (block_range * 1.1)
                    self.stop_losses.append(stop_loss)

                    take_profit = entry_price + ((entry_price - stop_loss) * RR)
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

    def double_top(self, RR):
        for x in range(4, self.new_df.shape[0]):
            if (
                pd.notna(self.new_df.iloc[x]["Is_Low"]) and
                pd.notna(self.new_df.iloc[x - 1]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 2]["Is_Low"]) and
                pd.notna(self.new_df.iloc[x - 3]["Is_High"]) and
                pd.notna(self.new_df.iloc[x - 4]["Is_Low"])
            ):
                tolerance = 0.05
                lower_tolerance = self.new_df.iloc[x - 3]["Is_High"] - self.new_df.iloc[x - 2]["Is_Low"] * math.tan(0.02)
                upper_tolerance = self.new_df.iloc[x - 3]["Is_High"] + self.new_df.iloc[x - 2]["Is_Low"] * math.tan(0.02)

                if (
                    (self.new_df.iloc[x]["Is_Low"] < self.new_df.iloc[x - 1]["Is_High"]) and
                    (self.new_df.iloc[x - 2]["Is_Low"] < self.new_df.iloc[x - 1]["Is_High"]) and
                    (self.new_df.iloc[x - 3]["Is_High"] > self.new_df.iloc[x - 2]["Is_Low"]) and
                    (self.new_df.iloc[x]["Is_Low"] < self.new_df.iloc[x - 2]["Is_Low"]) and
                    (self.new_df.iloc[x - 3]["Is_High"] < self.new_df.iloc[x - 1]["Is_High"]) and
                    (self.new_df.iloc[x - 4]["Is_Low"] < self.new_df.iloc[x - 2]["Is_Low"])
                ) and (
                    (self.new_df.iloc[x - 1]["Is_High"] - self.new_df.iloc[x]["Is_Low"]) >
                    (2 * (self.new_df.iloc[x - 1]["Is_High"] - self.new_df.iloc[x - 2]["Is_Low"]))
                ):
                    if upper_tolerance >= self.new_df.iloc[x - 1]["Is_High"] >= lower_tolerance:
                        self.occurences.append(self.new_df.index[x])
                        block_range = self.new_df.iloc[x - 3]["Is_High"] - self.new_df.iloc[x - 2]["Is_Low"]

                        entry_price = self.new_df.iloc[x - 2]["Is_Low"] - block_range
                        self.entries.append(entry_price)

                        stop_loss = self.new_df.iloc[x - 1]["Is_High"] + (block_range * self.fibonacci_levels[1])
                        self.stop_losses.append(stop_loss)

                        take_profit = entry_price - ((stop_loss - entry_price) * RR)
                        self.take_profits.append(take_profit)

        plot_df = pd.DataFrame({
            "Occurence": self.occurences,
            "Entry": self.entries,
            "Stop_Loss": self.stop_losses,
            "Take_Profit": self.take_profits,
            }
        )
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df


    def TripleTop(self, RR):
        # Breaker Block
        for x in range(self.new_df.shape[0]):
            if (
                (
                    pd.notna(self.new_df.iloc[x - 0]["Is_Low"])
                    and pd.notna(self.new_df.iloc[x - 1]["Is_High"])
                    and pd.notna(self.new_df.iloc[x - 2]["Is_Low"])
                    and pd.notna(self.new_df.iloc[x - 3]["Is_High"])
                    and pd.notna(self.new_df.iloc[x - 4]["Is_Low"])
                )
                and (
                    (self.new_df.iloc[x - 0]["Is_Low"]) < (self.new_df.iloc[x - 4]["Is_Low"])
                    and (self.new_df.iloc[x - 1]["Is_High"]) > (self.new_df.iloc[x - 2]["Is_Low"])
                    and (self.new_df.iloc[x - 3]["Is_High"]) > (self.new_df.iloc[x - 1]["Is_High"])
                    and (self.new_df.iloc[x - 4]["Is_Low"]) < (self.new_df.iloc[x - 2]["Is_Low"])
                    and (self.new_df.iloc[x - 5]["Is_High"]) < (self.new_df.iloc[x - 3]["Is_High"])
                    # and (self.new_df.iloc[x - 4]["Is_Low"]) > (self.new_df.iloc[x - 0]["Is_Low"])
                )
                # and (self.new_df.iloc[x - 1]["Is_High"]) - (self.new_df.iloc[x - 0]["Is_Low"])
                #     > (200/100 * (self.new_df.iloc[x - 1]["Is_High"] - self.new_df.iloc[x - 2]["Is_Low"]))
            ):

                # if (df.iloc[x-0,2] == 'low' and df.iloc[x-1,2] == 'high' and df.iloc[x-2,2] == 'low' and df.iloc[x-3,2] == 'high' and df.iloc[x-4,2] == 'low' and df.iloc[x-5,2] == 'high') and (df.iloc[x-0,1] < df.iloc[x-4,1]) and (df.iloc[x-1,1] > df.iloc[x-3,1]) and (df.iloc[x-3,1] > df.iloc[x-5,1]) and (df.iloc[x-5,1] > df.iloc[x-4,1]) and (df.iloc[x-4,1] < df.iloc[x-2,1]) and ((df.iloc[x-4,1] -df.iloc[x-0,1]) > 40/100 * (df.iloc[x-1,1] - df.iloc[x-2,1])):
                # TODO: Write code to check columns "Is_High" and "Is_Low" for where either Is_High or Is_Low is True, then check if Is_Low is True and Is_High is False and vice versa, then create a new dataframe with the results with the values of the dataframe df

                self.occurences.append(self.new_df.index[x - 0])
                block_range = (self.new_df.iloc[x - 1]["Is_High"]) - (self.new_df.iloc[x - 2]["Is_Low"])

                # entry_price = block_range * 0.236 + df.iloc[x - 2, 1]
                entry_price = self.new_df.iloc[x - 4]["Is_Low"]
                # (float((df.iloc[x-1,1] - df.iloc[x-2,1])/4) + df.iloc[x-2,1])
                self.entries.append(entry_price)

                # Calculate the stop loss and take profit levels
                stop_loss = (self.new_df.iloc[x - 1]["Is_High"]) + (
                    block_range * 0.05
                )  # 50% retracement from entry
                self.stop_losses.append(stop_loss)
                take_profit = entry_price - ((stop_loss - entry_price) * RR)
                self.take_profits.append(take_profit)
        # print(df)
        # print(self.occurences)
        # Create a DataFrame with the results
        plot_df = pd.DataFrame(
            {
                "Occurence": self.occurences,
                "Entry": self.entries,
                "Stop_Loss": self.stop_losses,
                "Take_Profit": self.take_profits,
            }
        )
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df


    def SMSstrategy(self, RR):
        # Calculate the moving average
        self.new_df['SMA'] = ta.sma(self.new_df['Close'], window=21)  # Adjust the window size as needed

        for x in range(self.new_df.shape[0]):
            if (
                (
                    pd.notna(self.new_df.iloc[x - 0]["Is_Low"])
                    and pd.notna(self.new_df.iloc[x - 1]["Is_High"])
                    and pd.notna(self.new_df.iloc[x - 2]["Is_Low"])
                    and pd.notna(self.new_df.iloc[x - 3]["Is_High"])
                    and pd.notna(self.new_df.iloc[x - 4]["Is_Low"])
                )
                and (
                    (self.new_df.iloc[x - 0]["Is_Low"]) > (self.new_df.iloc[x - 2]["Is_Low"])
                    and (self.new_df.iloc[x - 1]["Is_High"]) > (self.new_df.iloc[x - 2]["Is_Low"])
                    and (self.new_df.iloc[x - 1]["Is_High"]) > (self.new_df.iloc[x - 3]["Is_High"])
                    and (self.new_df.iloc[x - 3]["Is_High"]) > (self.new_df.iloc[x - 2]["Is_Low"])
                    and (self.new_df.iloc[x - 4]["Is_Low"]) < (self.new_df.iloc[x - 2]["Is_Low"])
                )
                and (self.new_df.iloc[x - 1]["High"] > self.new_df.iloc[x - 1]["SMA"])  # Entry signal based on SMA
            ):

                self.occurences.append(self.new_df.index[x - 0])
                block_range = (self.new_df.iloc[x - 3]["Is_High"]) - (self.new_df.iloc[x - 2]["Is_Low"])

                entry_price = self.new_df.iloc[x - 1]["Is_High"]
                self.entries.append(entry_price)

                stop_loss = (self.new_df.iloc[x - 1]["Is_High"]) - (
                    block_range * 1
                )  # 50% retracement from entry

                take_profit = entry_price - ((stop_loss - entry_price) * RR)

                self.stop_losses.append(stop_loss)
                self.take_profits.append(take_profit)

        plot_df = pd.DataFrame(
            {
                "Occurence": self.occurences,
                "Entry": self.entries,
                "Stop_Loss": self.stop_losses,
                "Take_Profit": self.take_profits,
            }
        )
        plot_df["Risk_to_Reward_Ratio"] = (plot_df["Take_Profit"] - plot_df["Entry"]) / (
            plot_df["Entry"] - plot_df["Stop_Loss"]
        )

        return plot_df