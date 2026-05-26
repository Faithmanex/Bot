import os
import sys
import json
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
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
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
        
        self.style.configure("TCombobox", 
            fieldbackground="#2A2A2A", 
            background="#2A2A2A", 
            foreground=self.colors["text"], 
            arrowcolor=self.colors["accent"],
            darkcolor="#1E1E1E",
            lightcolor="#2C2C2C",
            bordercolor="#2C2C2C"
        )
        self.style.map("TCombobox",
            fieldbackground=[("readonly", "#2A2A2A"), ("disabled", "#1A1A1A")],
            foreground=[("readonly", self.colors["text"]), ("disabled", "#666666")],
            background=[("readonly", "#2A2A2A")]
        )
        
        # Force Combobox pop-up dropdown list colors
        self.option_add("*TCombobox*Listbox.background", "#2A2A2A")
        self.option_add("*TCombobox*Listbox.foreground", self.colors["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", self.colors["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))

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

        tk.Label(
            selector_frame, text="  ←  →  keys to navigate",
            bg="#121212", fg=self.colors["text_muted"], font=("Segoe UI", 8)
        ).pack(side="left", padx=(12, 0))

        self.pattern_chart_frame = tk.Frame(self.patterns_container, bg="#121212")
        self.pattern_chart_frame.grid(row=1, column=0, sticky="nsew")
        self.pattern_chart_frame.grid_columnconfigure(0, weight=1)
        self.pattern_chart_frame.grid_rowconfigure(0, weight=1)

        # Initialize embedded Canvases and Metrics widgets
        self.create_placeholder_chart()
        self.create_metrics_dashboard()
        self.create_placeholder_pattern_chart()

        # Tab 4: Live Backtest Replay
        self.tab_replay = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.tab_replay, text="   LIVE REPLAY   ")
        self._build_replay_tab()

        # Core thread states
        self.stop_event = threading.Event()
        self.bot_thread = None
        self.backtest_trades = None
        self.sweep_symbol = None

        # Replay engine state
        self.replay_price_df  = None
        self.replay_trades_df = None
        self.replay_symbol    = ""
        self.replay_timeframe = ""
        self.replay_idx       = 0
        self.replay_trade_ptr = 0
        self.replay_revealed  = []
        self.replay_wins      = 0
        self.replay_losses    = 0
        self.replay_balance   = 1000.0
        self.replay_running   = False
        self.replay_after_id  = None

        # Keyboard navigation for Pattern Charts tab
        self.bind("<Left>",  self._navigate_trade)
        self.bind("<Right>", self._navigate_trade)

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

    def _navigate_trade(self, event):
        """Navigate trades with ← → keys when Pattern Charts tab is active."""
        try:
            active_tab = self.notebook.index(self.notebook.select())
            patterns_tab = self.notebook.index(self.tab_patterns)
        except Exception:
            return
        if active_tab != patterns_tab:
            return

        values = self.combo_trades["values"]
        if not values:
            return

        current = self.combo_trades.current()
        if event.keysym == "Right":
            new_idx = min(current + 1, len(values) - 1)
        else:
            new_idx = max(current - 1, 0)

        if new_idx != current:
            self.combo_trades.current(new_idx)
            self.plot_selected_trade()

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

            rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}
            df.rename(columns=rename_map, inplace=True)

            # Locate trigger candle
            try:
                trig_loc = df.index.get_loc(trig_time)
            except KeyError:
                trig_loc = int(np.abs(df.index - trig_time).argmin())

            start_pos = max(0, trig_loc - 60)
            end_pos   = min(len(df), trig_loc + 100)
            dfpl = df.iloc[start_pos:end_pos][["Open", "High", "Low", "Close", "Volume"]].copy()

            entry = float(trade["Entry"])
            sl    = float(trade["Stop_Loss"])
            tp    = float(trade["Take_Profit"])

            # Custom dark style matching the app palette
            mc = mpf.make_marketcolors(
                up="#00E676", down="#FF1744",
                edge="inherit", wick="inherit", volume="inherit"
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                facecolor="#121212",
                edgecolor="#2C2C2C",
                figcolor="#121212",
                gridcolor="#222222",
                gridstyle="--",
                rc={
                    "axes.labelcolor": "#888888",
                    "xtick.color": "#EEEEEE",
                    "ytick.color": "#EEEEEE",
                    "axes.titlecolor": "#00ADB5",
                }
            )

            # Build figure using returnfig so mplfinance owns the datetime x-axis
            # vlines/hlines use the same datetime index mplfinance plots — no positioning bug
            fig, axes = mpf.plot(
                dfpl,
                type="candle",
                style=style,
                title=f"\nPATTERN INSPECTOR \u00b7 {symbol}  ({trade['Result']})",
                warn_too_much_data=999999,
                vlines=dict(
                    vlines=[trig_time],
                    colors=["#FFD600"],
                    linewidths=[2],
                    linestyle="dotted",
                ),
                hlines=dict(
                    hlines=[sl, entry, tp],
                    colors=["#FF1744", "#00B0FF", "#00E676"],
                    linewidths=[1.5, 1.5, 1.5],
                    linestyle="dashed",
                ),
                returnfig=True,
                figsize=(8, 4.5),
            )

            fig.patch.set_facecolor("#121212")

            # Legend annotation
            ax0 = axes[0]
            ax0.annotate(f"SL  {sl:.5f}",   xy=(1, sl),    xycoords=("axes fraction", "data"), color="#FF1744", fontsize=7, ha="right", va="bottom")
            ax0.annotate(f"Entry  {entry:.5f}", xy=(1, entry), xycoords=("axes fraction", "data"), color="#00B0FF", fontsize=7, ha="right", va="bottom")
            ax0.annotate(f"TP  {tp:.5f}",    xy=(1, tp),    xycoords=("axes fraction", "data"), color="#00E676", fontsize=7, ha="right", va="bottom")

            # Destroy old canvas and embed the new figure
            for widget in self.pattern_chart_frame.winfo_children():
                widget.destroy()

            self.pattern_canvas = FigureCanvasTkAgg(fig, master=self.pattern_chart_frame)
            self.pattern_canvas.draw()
            self.pattern_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

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
        self.sweep_symbol = symbol
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

            if self.sweep_symbol:
                sym = self.sweep_symbol
                self.sweep_symbol = None
                self.after(200, lambda: self.show_sweep_selection(sym))
            else:
                # After backtest completes, render equity curve and load replay data
                self.notebook.select(self.tab_console)
                self.after(200, self.plot_equity_curve)
                sym = self.var_symbols.get().split(",")[0].strip()
                tf  = self.var_timeframe.get()
                self.after(400, lambda: self.load_replay_data(sym, tf))

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

    def show_sweep_selection(self, symbol):
        """Show a styled top-10 sweep result selection dialog."""
        top10_path = os.path.join(BACKTEST_SUMMARY_DIR, f"sweep_top10_{symbol}.json")
        if not os.path.exists(top10_path):
            self.log_message(f"[WARN] No sweep results file found for {symbol}.\n", "WARN")
            return

        with open(top10_path, "r") as fh:
            top10 = json.load(fh)

        if not top10:
            self.log_message("[WARN] Sweep produced no results to select from.\n", "WARN")
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Select Configuration to Save — {symbol}")
        dlg.configure(bg=self.colors["bg"])
        dlg.geometry("760x460")
        dlg.resizable(False, False)
        dlg.grab_set()

        # Header
        hdr = tk.Label(
            dlg, text=f"HYPER-SWEEP TOP-10 RESULTS  ·  {symbol}",
            bg=self.colors["bg"], fg=self.colors["accent"],
            font=("Segoe UI", 11, "bold")
        )
        hdr.pack(anchor="w", padx=20, pady=(16, 6))

        sub = tk.Label(
            dlg, text="Select a configuration and click SAVE to apply it to the MLPattern strategy.",
            bg=self.colors["bg"], fg=self.colors["text_muted"],
            font=("Segoe UI", 9)
        )
        sub.pack(anchor="w", padx=20, pady=(0, 12))

        # Column headers
        cols_frame = tk.Frame(dlg, bg="#1A1A1A")
        cols_frame.pack(fill="x", padx=20)
        for col, w in [("Rank", 5), ("RR", 10), ("Threshold", 12), ("Trades", 9), ("Wins", 7), ("Losses", 8), ("Win Rate", 10), ("Net Profit", 12)]:
            tk.Label(
                cols_frame, text=col,
                bg="#1A1A1A", fg=self.colors["accent"],
                font=("Segoe UI", 8, "bold"), width=w, anchor="w"
            ).pack(side="left", padx=3, pady=6)

        # Listbox
        lb_frame = tk.Frame(dlg, bg=self.colors["bg"])
        lb_frame.pack(fill="both", expand=True, padx=20, pady=(2, 0))

        scrollbar = tk.Scrollbar(lb_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        lb = tk.Listbox(
            lb_frame,
            yscrollcommand=scrollbar.set,
            bg="#1E1E1E",
            fg=self.colors["text"],
            selectbackground=self.colors["accent"],
            selectforeground="#FFFFFF",
            activestyle="none",
            font=("Consolas", 9),
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        lb.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=lb.yview)

        for i, r in enumerate(top10):
            profit_sign = "+" if r["NetProfit"] >= 0 else ""
            row_text = (
                f"  #{i+1:<4}  "
                f"RR {r['RR']:.1f}:1{'':<6}"
                f"Thr {r['Threshold']*100:.0f}%{'':<6}"
                f"Trades {int(r['Trades']):<6}"
                f"W {int(r['Wins']):<5}"
                f"L {int(r['Losses']):<5}"
                f"WR {r['WinRate']:.1f}%{'':<5}"
                f"Profit {profit_sign}${r['NetProfit']:,.2f}"
            )
            lb.insert(tk.END, row_text)
            if i == 0:
                lb.itemconfig(i, fg=self.colors["success"])

        lb.select_set(0)

        # Status label
        status_lbl = tk.Label(
            dlg, text="",
            bg=self.colors["bg"], fg=self.colors["success"],
            font=("Segoe UI", 9, "bold")
        )
        status_lbl.pack(pady=(8, 0))

        # Buttons
        btn_frame = tk.Frame(dlg, bg=self.colors["bg"])
        btn_frame.pack(pady=(4, 16))

        def on_save():
            sel = lb.curselection()
            if not sel:
                status_lbl.config(text="Select a row first.", fg=self.colors["warning"])
                return
            chosen = top10[sel[0]]
            from currency.find_best_pattern import save_sweep_result
            ok, msg = save_sweep_result(symbol, chosen)
            if ok:
                status_lbl.config(
                    text=f"✓ Saved  {msg}  for {symbol}  —  switch Strategy to MLPattern and run.",
                    fg=self.colors["success"]
                )
                save_btn.config(state="disabled")
                self.log_message(
                    f"[INFO] Sweep config saved for {symbol}: {msg}\n", "INFO"
                )
            else:
                status_lbl.config(text=f"✗ Save failed: {msg}", fg=self.colors["danger"])

        save_btn = ttk.Button(btn_frame, text="SAVE SELECTED CONFIG", command=on_save, style="Action.TButton")
        save_btn.pack(side="left", padx=(0, 10), ipadx=10)

        ttk.Button(btn_frame, text="CLOSE", command=dlg.destroy, style="Stop.TButton").pack(side="left", ipadx=10)

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE REPLAY ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def _build_replay_tab(self):
        self.tab_replay.grid_columnconfigure(0, weight=3)
        self.tab_replay.grid_columnconfigure(1, weight=1)
        self.tab_replay.grid_rowconfigure(0, weight=1)
        self.tab_replay.grid_rowconfigure(1, weight=0)

        # Candlestick chart area
        self.replay_chart_frame = tk.Frame(self.tab_replay, bg="#0D0D0D")
        self.replay_chart_frame.grid(row=0, column=0, sticky="nsew")
        self.replay_chart_frame.grid_columnconfigure(0, weight=1)
        self.replay_chart_frame.grid_rowconfigure(0, weight=1)

        # Live stats sidebar
        self.replay_stats_frame = tk.Frame(
            self.tab_replay, bg="#1E1E1E", width=220,
            highlightbackground="#2C2C2C", highlightthickness=1
        )
        self.replay_stats_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.replay_stats_frame.grid_propagate(False)
        self._build_replay_stats()

        # Controls bar
        ctrl = tk.Frame(self.tab_replay, bg="#141414", height=48)
        ctrl.grid(row=1, column=0, columnspan=2, sticky="ew")
        ctrl.grid_propagate(False)

        self.btn_replay_reset = ttk.Button(
            ctrl, text="◀◀  RESET", command=self.reset_replay, style="Stop.TButton"
        )
        self.btn_replay_reset.pack(side="left", padx=(12, 6), pady=10)

        self.btn_replay_play = ttk.Button(
            ctrl, text="▶  PLAY", command=self.toggle_replay, style="Action.TButton"
        )
        self.btn_replay_play.pack(side="left", padx=(0, 18), pady=10)

        tk.Label(ctrl, text="Speed:", bg="#141414", fg=self.colors["text_muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 4))

        self.replay_speed_var = tk.IntVar(value=1)
        for label, val in [("1×", 1), ("5×", 5), ("25×", 25), ("∞", 0)]:
            tk.Radiobutton(
                ctrl, text=label, variable=self.replay_speed_var, value=val,
                bg="#141414", fg=self.colors["text"],
                selectcolor=self.colors["accent"],
                activebackground="#141414", activeforeground=self.colors["accent"],
                font=("Segoe UI", 9, "bold"), indicatoron=False,
                relief="flat", bd=0, padx=10, pady=5,
                highlightthickness=0,
            ).pack(side="left", padx=2, pady=10)

        self.replay_pct_lbl = tk.Label(
            ctrl, text="0%", bg="#141414", fg=self.colors["accent"],
            font=("Segoe UI", 9, "bold")
        )
        self.replay_pct_lbl.pack(side="right", padx=(0, 4))

        self.replay_progress_lbl = tk.Label(
            ctrl, text="Candle 0 / 0  ", bg="#141414", fg=self.colors["text_muted"],
            font=("Segoe UI", 8)
        )
        self.replay_progress_lbl.pack(side="right")

        self._init_replay_figure()

    def _init_replay_figure(self):
        self.replay_fig = Figure(figsize=(8, 4.5), dpi=100, facecolor="#0D0D0D")
        self.replay_ax  = self.replay_fig.add_subplot(111)
        self.replay_ax.set_facecolor("#0D0D0D")
        self.replay_ax.set_title(
            "LIVE BACKTEST REPLAY  —  run a backtest to load data",
            color=self.colors["text_muted"], fontsize=9, weight="bold"
        )
        for sp in self.replay_ax.spines.values():
            sp.set_color("#222222")
        self.replay_ax.tick_params(colors="#555555", labelsize=7)
        self.replay_ax.grid(True, color="#191919", linestyle="--", linewidth=0.5)

        self.replay_canvas_widget = FigureCanvasTkAgg(self.replay_fig, master=self.replay_chart_frame)
        self.replay_canvas_widget.draw()
        self.replay_canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_replay_stats(self):
        tk.Label(
            self.replay_stats_frame, text="LIVE METRICS",
            bg="#1E1E1E", fg=self.colors["accent"], font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))

        self.replay_metric_widgets = {}

        def add_metric(label_text, key):
            f = tk.Frame(self.replay_stats_frame, bg="#1E1E1E")
            f.pack(fill="x", padx=15, pady=6)
            tk.Label(f, text=label_text, bg="#1E1E1E",
                     fg=self.colors["text_muted"], font=("Segoe UI", 8)).pack(anchor="w")
            lbl = tk.Label(f, text="--", bg="#1E1E1E",
                           fg=self.colors["text"], font=("Segoe UI", 11, "bold"))
            lbl.pack(anchor="w", pady=(1, 0))
            self.replay_metric_widgets[key] = lbl

        add_metric("Account Balance", "balance")
        add_metric("Net Profit / Loss", "profit")
        add_metric("Wins", "wins")
        add_metric("Losses", "losses")
        add_metric("Win Rate", "winrate")
        add_metric("Last Trade", "last_trade")

    # ── Data Loading ──────────────────────────────────────────────────────────

    def load_replay_data(self, symbol, timeframe_name):
        from currency.settings import HISTORY_DATA_DIR, BACKTEST_SUMMARY_DIR as BSD
        price_file  = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
        trades_file = os.path.join(BSD, f"detailed_results_{symbol}.csv")

        if not os.path.exists(price_file) or not os.path.exists(trades_file):
            return

        price_df = pd.read_csv(price_file)
        price_df["time"] = pd.to_datetime(price_df["time"])
        price_df.set_index("time", inplace=True)
        rename_map = {"open": "Open", "high": "High", "low": "Low",
                      "close": "Close", "tick_volume": "Volume"}
        price_df.rename(columns=rename_map, inplace=True)
        self.replay_price_df = price_df

        trades_df = pd.read_csv(trades_file)
        trades_df["Occurrence"] = pd.to_datetime(trades_df["Occurrence"])
        self.replay_trades_df = (
            trades_df[trades_df["Result"].isin(["TP", "SL"])]
            .sort_values("Occurrence")
            .reset_index(drop=True)
        )
        self.replay_symbol    = symbol
        self.replay_timeframe = timeframe_name
        self.reset_replay(render=True)
        self.log_message(
            f"[INFO] Live Replay loaded: {len(self.replay_price_df):,} candles, "
            f"{len(self.replay_trades_df)} closed trades for {symbol}.\n", "INFO"
        )

    # ── Playback Controls ─────────────────────────────────────────────────────

    def reset_replay(self, render=False):
        if self.replay_after_id:
            self.after_cancel(self.replay_after_id)
            self.replay_after_id = None
        self.replay_running   = False
        self.replay_idx       = 0
        self.replay_trade_ptr = 0
        self.replay_revealed  = []
        self.replay_wins      = 0
        self.replay_losses    = 0
        try:
            self.replay_balance = float(self.var_balance.get())
        except (ValueError, AttributeError):
            self.replay_balance = 1000.0
        self.btn_replay_play.config(text="▶  PLAY")
        if render and self.replay_price_df is not None:
            self._render_replay_frame()
            self._update_replay_stats()

    def toggle_replay(self):
        if self.replay_price_df is None:
            self.log_message("[WARN] Run a backtest first to enable live replay.\n", "WARN")
            return
        self.replay_running = not self.replay_running
        if self.replay_running:
            self.btn_replay_play.config(text="⏸  PAUSE")
            self._replay_tick()
        else:
            self.btn_replay_play.config(text="▶  PLAY")
            if self.replay_after_id:
                self.after_cancel(self.replay_after_id)
                self.replay_after_id = None

    # ── Tick / Step Logic ─────────────────────────────────────────────────────

    def _replay_tick(self):
        if not self.replay_running or self.replay_price_df is None:
            return

        total = len(self.replay_price_df)
        speed = self.replay_speed_var.get()

        if speed == 0:
            # ∞ mode: jump to the next trade occurrence
            if self.replay_trade_ptr < len(self.replay_trades_df):
                next_occ = self.replay_trades_df.iloc[self.replay_trade_ptr]["Occurrence"]
                loc = self.replay_price_df.index.searchsorted(next_occ)
                self.replay_idx = min(int(loc), total - 1)
            else:
                self.replay_idx = total - 1
            # Reveal all trades up to this point
            while self.replay_trade_ptr < len(self.replay_trades_df):
                t = self.replay_trades_df.iloc[self.replay_trade_ptr]
                if t["Occurrence"] <= self.replay_price_df.index[self.replay_idx]:
                    self._reveal_trade(t)
                    self.replay_trade_ptr += 1
                else:
                    break
        else:
            for _ in range(speed):
                if self.replay_idx >= total - 1:
                    break
                self.replay_idx += 1
                self._process_candle(self.replay_idx)

        self._render_replay_frame()
        self._update_replay_stats()

        if self.replay_idx >= total - 1:
            self.replay_running = False
            self.btn_replay_play.config(text="▶  PLAY")
            return

        delay = {1: 80, 5: 30, 25: 12, 0: 150}.get(speed, 30)
        self.replay_after_id = self.after(delay, self._replay_tick)

    def _process_candle(self, idx):
        candle_time = self.replay_price_df.index[idx]
        while self.replay_trade_ptr < len(self.replay_trades_df):
            t = self.replay_trades_df.iloc[self.replay_trade_ptr]
            if t["Occurrence"] <= candle_time:
                self._reveal_trade(t)
                self.replay_trade_ptr += 1
            else:
                break

    def _reveal_trade(self, trade_row):
        self.replay_revealed.append(trade_row.to_dict())
        self.replay_balance = float(trade_row["Balance"])
        if trade_row["Result"] == "TP":
            self.replay_wins += 1
        else:
            self.replay_losses += 1

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_replay_frame(self):
        if self.replay_price_df is None:
            return

        WINDOW = 130
        total   = len(self.replay_price_df)
        end_idx = self.replay_idx + 1
        start_idx = max(0, end_idx - WINDOW)
        df_win = self.replay_price_df.iloc[start_idx:end_idx]

        ax = self.replay_ax
        ax.clear()
        ax.set_facecolor("#0D0D0D")
        for sp in ax.spines.values():
            sp.set_color("#222222")
        ax.grid(True, color="#161616", linestyle="--", linewidth=0.4)
        ax.tick_params(colors="#555555", labelsize=7)

        xs     = np.arange(len(df_win))
        opens  = df_win["Open"].values
        highs  = df_win["High"].values
        lows   = df_win["Low"].values
        closes = df_win["Close"].values
        is_up  = closes >= opens

        up_col   = "#00E676"
        down_col = "#FF1744"

        # Vectorised wicks
        wick_segs   = [[(x, lows[i]), (x, highs[i])] for i, x in enumerate(xs)]
        wick_colors = [up_col if is_up[i] else down_col for i in range(len(xs))]
        ax.add_collection(LineCollection(wick_segs, colors=wick_colors,
                                         linewidths=0.9, zorder=1))

        # Vectorised bodies
        bodies       = []
        body_colors  = []
        for i, x in enumerate(xs):
            bot  = min(opens[i], closes[i])
            h    = max(abs(closes[i] - opens[i]), (highs[i] - lows[i]) * 0.008)
            bodies.append(Rectangle((x - 0.38, bot), 0.76, h))
            body_colors.append(up_col if is_up[i] else down_col)

        pc = PatchCollection(bodies, facecolors=body_colors,
                             edgecolors=body_colors, linewidths=0.3, zorder=2)
        ax.add_collection(pc)

        # Trade markers and lines for trades visible in this window
        win_times = df_win.index
        if len(win_times) == 0:
            ax.set_xlim(-1, WINDOW)
            self.replay_canvas_widget.draw()
            return

        last_visible = None
        for trade in self.replay_revealed:
            occ = pd.to_datetime(trade["Occurrence"])
            if occ < win_times[0] or occ > win_times[-1]:
                continue
            pos = int(win_times.searchsorted(occ))
            pos = min(pos, len(df_win) - 1)
            result = trade["Result"]
            if result == "TP":
                ax.scatter(xs[pos], lows[pos] * 0.9999, marker="^",
                           color=up_col, s=55, zorder=6)
            else:
                ax.scatter(xs[pos], highs[pos] * 1.0001, marker="v",
                           color=down_col, s=55, zorder=6)
            last_visible = trade

        # SL / Entry / TP lines for the most recent visible trade
        if last_visible:
            entry = float(last_visible["Entry"])
            sl    = float(last_visible["Stop_Loss"])
            tp    = float(last_visible["Take_Profit"])
            ax.axhline(entry, color="#00B0FF", linestyle="--", linewidth=1.0,
                       alpha=0.65, zorder=3)
            ax.axhline(sl,    color=down_col,  linestyle="--", linewidth=0.9,
                       alpha=0.55, zorder=3)
            ax.axhline(tp,    color=up_col,    linestyle="--", linewidth=0.9,
                       alpha=0.55, zorder=3)

        ax.set_xlim(-1, WINDOW)
        ax.autoscale_view(scalex=False)

        # Title with live P&L
        try:
            init_bal = float(self.var_balance.get())
        except (ValueError, AttributeError):
            init_bal = 1000.0
        profit = self.replay_balance - init_bal
        sign   = "+" if profit >= 0 else ""
        p_col  = up_col if profit >= 0 else down_col
        ax.set_title(
            f"LIVE REPLAY  ·  {self.replay_symbol} ({self.replay_timeframe})     "
            f"Balance: ${self.replay_balance:,.2f}   [{sign}${profit:,.2f}]",
            color=p_col, fontsize=9, weight="bold", pad=6
        )

        self.replay_canvas_widget.draw()

        pct = int(end_idx / total * 100) if total > 0 else 0
        self.replay_progress_lbl.config(text=f"Candle {end_idx:,} / {total:,}  ")
        self.replay_pct_lbl.config(text=f"{pct}%")

    def _update_replay_stats(self):
        try:
            init_bal = float(self.var_balance.get())
        except (ValueError, AttributeError):
            init_bal = 1000.0

        profit = self.replay_balance - init_bal
        total  = self.replay_wins + self.replay_losses
        wr     = (self.replay_wins / total * 100) if total > 0 else 0.0

        p_text  = f"+${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}"
        p_color = self.colors["success"] if profit >= 0 else self.colors["danger"]

        last_text  = "--"
        last_color = self.colors["text_muted"]
        if self.replay_revealed:
            last = self.replay_revealed[-1]
            last_text  = f"{last['Result']} @ {float(last['Entry']):.5f}"
            last_color = (self.colors["success"] if last["Result"] == "TP"
                          else self.colors["danger"])

        self.replay_metric_widgets["balance"].config(
            text=f"${self.replay_balance:,.2f}", fg=self.colors["text"])
        self.replay_metric_widgets["profit"].config(text=p_text, fg=p_color)
        self.replay_metric_widgets["wins"].config(
            text=str(self.replay_wins), fg=self.colors["success"])
        self.replay_metric_widgets["losses"].config(
            text=str(self.replay_losses), fg=self.colors["danger"])
        self.replay_metric_widgets["winrate"].config(
            text=f"{wr:.1f}%", fg=self.colors["accent"])
        self.replay_metric_widgets["last_trade"].config(
            text=last_text, fg=last_color)


if __name__ == "__main__":
    app = TradingBotGUI()
    app.mainloop()
