import os
import sys
import threading
from io import StringIO
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import mplfinance as mpf

from currency.unified_trading import main as run_trading_logic
from currency.modules import trading_pairs
from currency.settings import BACKTEST_SUMMARY_DIR

class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Echelnet Unified Algorithmic Trading Terminal")
        self.geometry("1200x750")
        self.configure(bg="#121212")

        # Premium Dark Palette
        self.colors = {
            "bg": "#121212",
            "card_bg": "#1E1E1E",
            "accent": "#00ADB5",
            "accent_hover": "#00F5FF",
            "text": "#EEEEEE",
            "text_muted": "#888888",
            "success": "#00E676",
            "danger": "#FF1744",
            "warning": "#FFD600",
            "border": "#2C2C2C"
        }

        # TTK Style Customizations
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Global widget overrides
        self.style.configure(".", background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("Card.TFrame", background=self.colors["card_bg"], relief="flat", borderwidth=0)
        
        # Label designs
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground=self.colors["accent"])
        self.style.configure("CardHeader.TLabel", background=self.colors["card_bg"], font=("Segoe UI", 10, "bold"), foreground=self.colors["accent"])
        
        # Tabs Style (Notebook override)
        self.style.configure("TNotebook", background=self.colors["bg"], borderwidth=0, highlightthickness=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 9, "bold"), padding=(15, 6), background="#1A1A1A", foreground=self.colors["text_muted"])
        self.style.map("TNotebook.Tab",
            background=[("selected", self.colors["card_bg"]), ("active", "#252525")],
            foreground=[("selected", self.colors["accent"]), ("active", self.colors["text"])]
        )

        # Standard Premium Buttons
        self.style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), foreground="#FFFFFF", background=self.colors["accent"], borderwidth=0, padding=7)
        self.style.map("Action.TButton",
            background=[("active", self.colors["accent_hover"]), ("disabled", "#333333")],
            foreground=[("disabled", "#666666")]
        )
        
        self.style.configure("Stop.TButton", font=("Segoe UI", 9, "bold"), foreground="#FFFFFF", background=self.colors["danger"], borderwidth=0, padding=7)
        self.style.map("Stop.TButton",
            background=[("active", "#FF5252"), ("disabled", "#333333")],
            foreground=[("disabled", "#666666")]
        )

        # Combobox & Radio overrides
        self.style.configure("TRadiobutton", background=self.colors["card_bg"], foreground=self.colors["text"], font=("Segoe UI", 9))
        self.style.configure("TCombobox", fieldbackground="#2A2A2A", background="#2A2A2A", foreground=self.colors["text"], arrowcolor=self.colors["accent"])

        # Grid weights setup
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Left Column Panel (Controls & Form)
        self.left_panel = ttk.Frame(self, style="TFrame")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Title Block
        self.title_label = ttk.Label(self.left_panel, text="ECHELNET TRADING ENGINE", style="Header.TLabel")
        self.title_label.pack(anchor="w", pady=(0, 10))

        # Config Panel Card
        self.config_card = ttk.Frame(self.left_panel, style="Card.TFrame")
        self.config_card.pack(fill="both", expand=True, pady=(0, 12))
        self.create_config_form()

        # Operational Controls Card
        self.control_card = ttk.Frame(self.left_panel, style="Card.TFrame")
        self.control_card.pack(fill="x", pady=0)
        self.create_control_block()

        # Right Column Panel (Notebook Tabs for Terminal / Graphic Chart)
        self.right_panel = ttk.Frame(self, style="TFrame")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 15), pady=15)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

        # Dashboard Tab Controller
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tab 1: Terminal Log Console
        self.tab_console = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.tab_console, text="   LOG TERMINAL   ")
        self.tab_console.grid_columnconfigure(0, weight=1)
        self.tab_console.grid_rowconfigure(0, weight=1)

        self.log_area = scrolledtext.ScrolledText(
            self.tab_console, 
            wrap=tk.WORD, 
            bg="#0B0B0B", 
            fg="#00FF66", 
            insertbackground="#EEEEEE",
            font=("Consolas", 9),
            bd=0,
            highlightthickness=0
        )
        self.log_area.grid(row=0, column=0, sticky="nsew")

        # Tab 2: Visual Equity Growth Chart & Metrics Dashboard Panel
        self.tab_chart = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.tab_chart, text="   EQUITY GRAPH   ")
        self.tab_chart.grid_columnconfigure(0, weight=3)  # Matplotlib curve panel
        self.tab_chart.grid_columnconfigure(1, weight=1)  # Quantitative stats panel
        self.tab_chart.grid_rowconfigure(0, weight=1)

        self.chart_frame = tk.Frame(self.tab_chart, bg="#121212")
        self.chart_frame.grid(row=0, column=0, sticky="nsew")
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_frame.grid_rowconfigure(0, weight=1)

        self.metrics_frame = tk.Frame(self.tab_chart, bg="#1E1E1E", width=260, highlightbackground="#2C2C2C", highlightthickness=1)
        self.metrics_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        self.metrics_frame.grid_propagate(False)

        # Tab 3: Visual Candlestick Trade Pattern Inspector
        self.tab_patterns = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.tab_patterns, text="   PATTERN CHARTS   ")
        self.tab_patterns.grid_columnconfigure(0, weight=1)
        self.tab_patterns.grid_rowconfigure(0, weight=1)
        
        self.patterns_container = tk.Frame(self.tab_patterns, bg="#121212")
        self.patterns_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.patterns_container.grid_columnconfigure(0, weight=1)
        self.patterns_container.grid_rowconfigure(1, weight=1)

        # Trade selector combobox
        selector_frame = tk.Frame(self.patterns_container, bg="#121212")
        selector_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        lbl_select = tk.Label(selector_frame, text="Select Trade Setup:", bg="#121212", fg=self.colors["accent"], font=("Segoe UI", 9, "bold"))
        lbl_select.pack(side="left", padx=(0, 10))
        
        self.combo_trades = ttk.Combobox(selector_frame, state="readonly", width=55)
        self.combo_trades.pack(side="left")
        self.combo_trades.bind("<<ComboboxSelected>>", self.plot_selected_trade)

        self.pattern_chart_frame = tk.Frame(self.patterns_container, bg="#121212")
        self.pattern_chart_frame.grid(row=1, column=0, sticky="nsew")
        self.pattern_chart_frame.grid_columnconfigure(0, weight=1)
        self.pattern_chart_frame.grid_rowconfigure(0, weight=1)

        # Initialize embedded Canvases and Metrics widgets
        self.create_placeholder_chart()
        self.create_metrics_dashboard()
        self.create_placeholder_pattern_chart()

        # Core thread states
        self.stop_event = threading.Event()
        self.bot_thread = None
        self.backtest_trades = None

    def create_config_form(self):
        lbl = tk.Label(self.config_card, text="STRATEGY PARAMETERS", bg=self.colors["card_bg"], fg=self.colors["accent"], font=("Segoe UI", 10, "bold"))
        lbl.pack(anchor="w", padx=15, pady=(12, 8))

        form_frame = tk.Frame(self.config_card, bg=self.colors["card_bg"])
        form_frame.pack(fill="both", expand=True, padx=15, pady=0)
        form_frame.columnconfigure(1, weight=1)

        def add_field(row, label_text, var_type="entry", default_val="", options=None):
            lbl_w = tk.Label(form_frame, text=label_text, bg=self.colors["card_bg"], fg=self.colors["text"], font=("Segoe UI", 9))
            lbl_w.grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))
            
            if var_type == "entry":
                var = tk.StringVar(value=default_val)
                entry = tk.Entry(form_frame, textvariable=var, bg="#2A2A2A", fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, relief="flat", highlightthickness=1, highlightbackground="#3A3A3A", highlightcolor=self.colors["accent"], font=("Segoe UI", 9))
                entry.grid(row=row, column=1, sticky="ew", pady=4)
                return var
            elif var_type == "combo":
                var = tk.StringVar(value=default_val)
                combo = ttk.Combobox(form_frame, textvariable=var, values=options, state="readonly")
                combo.grid(row=row, column=1, sticky="ew", pady=4)
                return var

        default_pairs = ", ".join(trading_pairs.symbols)
        self.var_symbols = add_field(0, "Symbols (comma list):", "entry", default_pairs)
        self.var_timeframe = add_field(1, "Timeframe:", "combo", "M5", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        
        two_months_ago = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        self.var_start_date = add_field(2, "Start Date (YYYY-MM-DD):", "entry", two_months_ago)
        self.var_end_date = add_field(3, "End Date (YYYY-MM-DD):", "entry", datetime.now().strftime("%Y-%m-%d"))
        
        self.var_strategy = add_field(4, "Strategy Model:", "combo", "Noir", ["Noir", "BreakerBlock", "DoubleTop", "TripleTop", "MLPattern"])
        self.var_balance = add_field(5, "Initial Balance ($):", "entry", "1000.0")
        self.var_risk_amount = add_field(6, "Risk Value:", "entry", "25.0")
        self.var_risk_type = add_field(7, "Risk Metric Type:", "combo", "fixed", ["fixed", "percentage"])
        self.var_rr = add_field(8, "Risk-to-Reward (RR):", "entry", "5.0")

    def create_control_block(self):
        lbl = tk.Label(self.control_card, text="OPERATIONAL CONTROLS", bg=self.colors["card_bg"], fg=self.colors["accent"], font=("Segoe UI", 10, "bold"))
        lbl.pack(anchor="w", padx=15, pady=(12, 8))

        btn_frame = tk.Frame(self.control_card, bg=self.colors["card_bg"])
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.mode_var = tk.StringVar(value="backtest")
        
        rb_backtest = ttk.Radiobutton(btn_frame, text="Run Backtest Simulation", variable=self.mode_var, value="backtest")
        rb_backtest.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        rb_live = ttk.Radiobutton(btn_frame, text="Execute Live Market Trading", variable=self.mode_var, value="live")
        rb_live.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.start_button = ttk.Button(btn_frame, text="START BOT ENGINE", command=self.start_bot, style="Action.TButton")
        self.start_button.grid(row=2, column=0, sticky="ew", padx=(0, 4))

        self.stop_button = ttk.Button(btn_frame, text="FORCE SHUTDOWN", command=self.stop_bot, style="Stop.TButton", state="disabled")
        self.stop_button.grid(row=2, column=1, sticky="ew", padx=(4, 0))

        self.sweep_button = ttk.Button(btn_frame, text="RUN HYPER-SWEEP OPTIMIZER", command=self.start_sweep, style="Action.TButton")
        self.sweep_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def create_placeholder_chart(self):
        # Create gorgeous matching dark themed canvas
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor="#121212")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#121212")
        
        self.ax.tick_params(colors="#EEEEEE", labelsize=8)
        self.ax.spines['bottom'].set_color('#2C2C2C')
        self.ax.spines['top'].set_color('#2C2C2C')
        self.ax.spines['left'].set_color('#2C2C2C')
        self.ax.spines['right'].set_color('#2C2C2C')
        self.ax.grid(True, color="#222222", linestyle="--")
        
        self.ax.set_title("ACCOUNT EQUITY CURVE", color="#00ADB5", fontname="Segoe UI", fontsize=10, weight="bold")
        self.ax.set_xlabel("Number of Trades", color="#888888", fontname="Segoe UI", fontsize=8)
        self.ax.set_ylabel("Account Balance ($)", color="#888888", fontname="Segoe UI", fontsize=8)
        self.ax.plot([], [], color="#00ADB5")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def create_placeholder_pattern_chart(self):
        # Placeholder figure for detailed trade patterns
        self.pattern_fig = Figure(figsize=(6, 4), dpi=100, facecolor="#121212")
        self.pattern_ax = self.pattern_fig.add_subplot(111)
        self.pattern_ax.set_facecolor("#121212")
        
        self.pattern_ax.tick_params(colors="#EEEEEE", labelsize=8)
        self.pattern_ax.spines['bottom'].set_color('#2C2C2C')
        self.pattern_ax.spines['top'].set_color('#2C2C2C')
        self.pattern_ax.spines['left'].set_color('#2C2C2C')
        self.pattern_ax.spines['right'].set_color('#2C2C2C')
        self.pattern_ax.grid(True, color="#222222", linestyle="--")
        
        self.pattern_ax.set_title("CANDLESTICK TRADE PLOT", color="#00ADB5", fontname="Segoe UI", fontsize=10, weight="bold")
        self.pattern_ax.set_xlabel("Timeframe Intervals", color="#888888", fontname="Segoe UI", fontsize=8)
        self.pattern_ax.set_ylabel("Asset Price", color="#888888", fontname="Segoe UI", fontsize=8)
        
        self.pattern_canvas = FigureCanvasTkAgg(self.pattern_fig, master=self.pattern_chart_frame)
        self.pattern_canvas.draw()
        self.pattern_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def create_metrics_dashboard(self):
        # Header
        lbl = tk.Label(self.metrics_frame, text="TERMINAL METRICS", bg=self.colors["card_bg"], fg=self.colors["accent"], font=("Segoe UI", 10, "bold"))
        lbl.pack(anchor="w", padx=15, pady=(15, 8))
        
        self.metric_widgets = {}
        
        def add_metric(label_text, key):
            frame = tk.Frame(self.metrics_frame, bg=self.colors["card_bg"])
            frame.pack(fill="x", padx=15, pady=5)
            
            lbl_name = tk.Label(frame, text=label_text, bg=self.colors["card_bg"], fg=self.colors["text_muted"], font=("Segoe UI", 8))
            lbl_name.pack(anchor="w")
            
            lbl_val = tk.Label(frame, text="--", bg=self.colors["card_bg"], fg=self.colors["text"], font=("Segoe UI", 11, "bold"))
            lbl_val.pack(anchor="w", pady=(1, 0))
            
            self.metric_widgets[key] = lbl_val

        add_metric("Net Profit / Loss", "profit")
        add_metric("Overall Win Rate", "winrate")
        add_metric("Profit Factor", "profit_factor")
        add_metric("Total Executed Trades", "total_trades")
        add_metric("Max Account Drawdown", "drawdown")
        add_metric("Consecutive Loss Streak", "loss_streak")

    def plot_equity_curve(self):
        symbol = self.var_symbols.get().split(",")[0].strip()
        detailed_file = os.path.join(BACKTEST_SUMMARY_DIR, f"detailed_results_{symbol}.csv")
        
        if not os.path.exists(detailed_file):
            return
            
        try:
            df_results = pd.read_csv(detailed_file)
            if df_results.empty or "Balance" not in df_results.columns:
                return
                
            balances = df_results["Balance"].to_numpy()
            init_bal = float(self.var_balance.get().strip())
            balances = np.insert(balances, 0, init_bal)
            
            # 1. Clear and repaint Matplotlib canvas
            self.ax.clear()
            self.ax.set_facecolor("#121212")
            self.ax.tick_params(colors="#EEEEEE", labelsize=8)
            self.ax.spines['bottom'].set_color('#2C2C2C')
            self.ax.spines['top'].set_color('#2C2C2C')
            self.ax.spines['left'].set_color('#2C2C2C')
            self.ax.spines['right'].set_color('#2C2C2C')
            self.ax.grid(True, color="#222222", linestyle="--")
            
            self.ax.set_title(f"ACCOUNT EQUITY GROWTH: {symbol}", color="#00ADB5", fontname="Segoe UI", fontsize=10, weight="bold")
            self.ax.set_xlabel("Number of Trades", color="#888888", fontname="Segoe UI", fontsize=8)
            self.ax.set_ylabel("Account Balance ($)", color="#888888", fontname="Segoe UI", fontsize=8)
            
            glow_color = self.colors["success"] if balances[-1] >= init_bal else self.colors["danger"]
            x = np.arange(len(balances))
            self.ax.plot(x, balances, color=glow_color, linewidth=2)
            self.ax.fill_between(x, balances, init_bal, color=glow_color, alpha=0.1)
            
            self.canvas.draw()

            # 2. Calculate Quantitative Statistics & populate Sidebar Cards
            net_profit = balances[-1] - init_bal
            total_trades = len(df_results)
            
            closed_trades = df_results[df_results["Result"].isin(["SL", "TP"])]
            wins = len(closed_trades[closed_trades["Result"] == "TP"])
            losses = len(closed_trades[closed_trades["Result"] == "SL"])
            total_closed = wins + losses
            
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
            
            # Profit Factor
            gross_profit = wins * (float(self.var_risk_amount.get()) * float(self.var_rr.get()))
            gross_loss = losses * float(self.var_risk_amount.get())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
            
            # Max Drawdown
            peak = balances[0]
            max_dd_pct = 0.0
            for b in balances:
                if b > peak:
                    peak = b
                dd = (peak - b) / peak if peak > 0 else 0.0
                if dd > max_dd_pct:
                    max_dd_pct = dd
            max_dd_val = max_dd_pct * 100
            
            # Streak
            current_streak = 0
            max_streak = 0
            for res in df_results["Result"]:
                if res == "SL":
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                elif res == "TP":
                    current_streak = 0

            # Update Metrics Panel Labels dynamically
            profit_widget = self.metric_widgets["profit"]
            if net_profit >= 0:
                profit_widget.config(text=f"+${net_profit:,.2f}", fg=self.colors["success"])
            else:
                profit_widget.config(text=f"-${abs(net_profit):,.2f}", fg=self.colors["danger"])
                
            self.metric_widgets["winrate"].config(text=f"{win_rate:.1f}%", fg=self.colors["accent"])
            self.metric_widgets["profit_factor"].config(text=f"{profit_factor:.2f}", fg=self.colors["warning"])
            self.metric_widgets["total_trades"].config(text=f"{total_trades}", fg=self.colors["text"])
            self.metric_widgets["drawdown"].config(text=f"{max_dd_val:.1f}%", fg=self.colors["danger"])
            self.metric_widgets["loss_streak"].config(text=f"{max_streak} SL", fg=self.colors["text_muted"])
            
            # 3. Populate Trade Selector Dropdown for Pattern Tab
            self.backtest_trades = df_results[df_results["Result"].isin(["TP", "SL"])].copy()
            trade_options = []
            for i, row in self.backtest_trades.iterrows():
                trade_options.append(f"#{i+1}: {row['Occurrence']} | {row['Result']} @ {row['Entry']:.5f}")
            
            self.combo_trades.config(values=trade_options)
            if trade_options:
                self.combo_trades.current(0)
                self.plot_selected_trade()

            # Auto switch tab to show off the visual chart!
            self.notebook.select(self.tab_chart)
            
        except Exception as e:
            print(f"[ERROR] Failed to render equity curve: {e}")

    def plot_selected_trade(self, event=None):
        symbol = self.var_symbols.get().split(",")[0].strip()
        timeframe_name = self.var_timeframe.get()
        
        sel_idx = self.combo_trades.current()
        if sel_idx < 0 or self.backtest_trades is None or self.backtest_trades.empty:
            return
            
        trade = self.backtest_trades.iloc[sel_idx]
        trig_time = pd.to_datetime(trade["Occurrence"])
        
        from currency.settings import HISTORY_DATA_DIR
        filename = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
        if not os.path.exists(filename):
            return
            
        try:
            df = pd.read_csv(filename)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            
            # Casing correction helper
            rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}
            df.rename(columns=rename_map, inplace=True)
            
            # Slicing the sub-window (-15, +35 candles) around trigger point
            try:
                trig_loc = df.index.get_loc(trig_time)
            except KeyError:
                # Find nearest match if exact key isn't present
                trig_loc = np.abs(df.index - trig_time).argmin()

            start_pos = max(0, trig_loc - 15)
            end_pos = min(len(df), trig_loc + 35)
            dfpl = df.iloc[start_pos:end_pos].copy()
            
            # Reset visual plot
            self.pattern_ax.clear()
            self.pattern_ax.set_facecolor("#121212")
            
            # Plot Candlesticks using native mpf inside embedded ax
            mpf.plot(
                dfpl,
                type="candle",
                ax=self.pattern_ax,
                style="charles",
                warn_too_much_data=999999
            )
            
            # Draw Horizontal Price lines
            entry = float(trade["Entry"])
            sl = float(trade["Stop_Loss"])
            tp = float(trade["Take_Profit"])
            
            self.pattern_ax.axhline(entry, color="#00B0FF", linestyle="--", linewidth=1.5, label=f"Entry: {entry:.5f}")
            self.pattern_ax.axhline(sl, color="#FF1744", linestyle="--", linewidth=1.5, label=f"SL: {sl:.5f}")
            self.pattern_ax.axhline(tp, color="#00E676", linestyle="--", linewidth=1.5, label=f"TP: {tp:.5f}")
            
            # Draw Vertical Trigger marker
            self.pattern_ax.axvline(dfpl.index[min(len(dfpl)-1, max(0, trig_loc - start_pos))], color="#FFD600", linestyle=":", linewidth=2, label="Setup Trigger")
            
            # Display Stylized Legend and titles
            self.pattern_ax.legend(facecolor="#1E1E1E", edgecolor="#2C2C2C", labelcolor="#EEEEEE", loc="best", fontsize=8)
            self.pattern_ax.set_title(f"PATTERN GRAPH INSPECTOR: {symbol} ({trade['Result']})", color="#00ADB5", fontname="Segoe UI", fontsize=10, weight="bold")
            
            self.pattern_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] Failed to draw trade setup chart: {e}")

    def start_bot(self):
        try:
            start_dt = datetime.strptime(self.var_start_date.get().strip(), "%Y-%m-%d")
            end_dt = datetime.strptime(self.var_end_date.get().strip(), "%Y-%m-%d")
            if start_dt > end_dt:
                self.log_message("[ERROR] Start Date must be BEFORE End Date! Please verify your timeline range.\n", "ERROR")
                return
            init_bal = float(self.var_balance.get().strip())
            risk_val = float(self.var_risk_amount.get().strip())
            rr_val = float(self.var_rr.get().strip())
        except ValueError as e:
            self.log_message(f"[ERROR] Invalid format: {e}\n", "ERROR")
            return

        live_trading = self.mode_var.get() == "live"
        self.start_button.config(state="disabled")
        self.sweep_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.log_area.delete("1.0", tk.END)
        self.log_message(f"[INFO] Initializing system in {'Live Trading' if live_trading else 'Backtest'} mode...\n", "INFO")
        
        config = {
            "symbols": self.var_symbols.get(),
            "timeframe": self.var_timeframe.get(),
            "start_time": start_dt,
            "end_time": end_dt,
            "strategy": self.var_strategy.get(),
            "initial_balance": init_bal,
            "risk_amount": risk_val,
            "risk_type": self.var_risk_type.get(),
            "rr": rr_val
        }

        self.stop_event.clear()
        
        self.log_stream = StringIO()
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream

        # Spawn Engine Thread
        self.bot_thread = threading.Thread(target=self.run_bot_logic, args=(live_trading, config), daemon=True)
        self.bot_thread.start()
        self.after(100, self.update_logs)

    def run_bot_logic(self, live_trading, config):
        try:
            run_trading_logic(live_trading, self.stop_event, config)
        except Exception as e:
            self.log_message(f"\n[ERROR] Thread execution failed: {e}\n", "ERROR")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def start_sweep(self):
        symbol = self.var_symbols.get().split(",")[0].strip()
        self.start_button.config(state="disabled")
        self.sweep_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.log_area.delete("1.0", tk.END)
        self.log_message(f"[INFO] Initializing hyper-sweep parameter optimizer for {symbol}...\n", "INFO")
        self.log_message("[INFO] Model loading has been fully cached. Scan will be extremely rapid.\n", "INFO")
        
        self.stop_event.clear()
        
        self.log_stream = StringIO()
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream

        self.bot_thread = threading.Thread(target=self.run_sweep_logic, args=(symbol,), daemon=True)
        self.bot_thread.start()
        self.after(100, self.update_logs)

    def run_sweep_logic(self, symbol):
        try:
            from currency.find_best_pattern import run_sweep
            run_sweep(symbol=symbol)
        except Exception as e:
            self.log_message(f"\n[ERROR] Sweep thread execution failed: {e}\n", "ERROR")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def update_logs(self):
        log_contents = self.log_stream.getvalue()
        if log_contents:
            for line in log_contents.splitlines(keepends=True):
                tag = None
                if "SL" in line or "losses" in line:
                    tag = "SL"
                elif "TP" in line or "Wins" in line:
                    tag = "TP"
                elif "Error" in line or "failed" in line or "Invalid params" in line or "[ERROR]" in line:
                    tag = "ERROR"
                elif "Skipping" in line or "Warning" in line:
                    tag = "WARN"
                elif "Initializing" in line or "info" in line or "[INFO]" in line:
                    tag = "INFO"
                self.log_message(line, tag)
            
            self.log_stream.seek(0)
            self.log_stream.truncate(0)

        if self.bot_thread and self.bot_thread.is_alive():
            self.after(100, self.update_logs)
        else:
            self.start_button.config(state="normal")
            self.sweep_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.log_message("\n[INFO] Engine process terminated.\n", "INFO")
            
            # After backtest completes, dynamically render the Equity curve chart!
            self.notebook.select(self.tab_console)  # Keep console tab focus unless chart renders successfully
            self.after(200, self.plot_equity_curve)

    def stop_bot(self):
        self.stop_event.set()
        self.log_message("\n[WARN] Shutdown request issued. Halting gracefully...\n", "WARN")
        self.stop_button.config(state="disabled")

    def log_message(self, message, tag=None):
        self.log_area.config(state="normal")
        if tag:
            self.log_area.insert(tk.END, message, tag)
        else:
            self.log_area.insert(tk.END, message)
        self.log_area.config(state="disabled")
        self.log_area.see(tk.END)

if __name__ == "__main__":
    app = TradingBotGUI()
    app.mainloop()
