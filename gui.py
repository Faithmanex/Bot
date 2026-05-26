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

import webbrowser
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
        lbl_select.pack(side="left", padx=(0, 8))

        lbl_filter = tk.Label(selector_frame, text="Filter:", bg="#121212", fg=self.colors["accent"], font=("Segoe UI", 9, "bold"))
        lbl_filter.pack(side="left", padx=(10, 4))

        self.combo_filter = ttk.Combobox(selector_frame, state="readonly", values=["All", "Buys", "Sells"], width=7)
        self.combo_filter.set("All")
        self.combo_filter.pack(side="left", padx=(0, 12))
        self.combo_filter.bind("<<ComboboxSelected>>", self.apply_gui_trade_filter)

        self.combo_trades = ttk.Combobox(selector_frame, state="readonly", width=45)
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

        # Tab 5: ML Training
        self.tab_ml_train = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.tab_ml_train, text="   ML TRAINING   ")
        self._build_ml_train_tab()

        # Core thread states
        self.stop_event = threading.Event()
        self.bot_thread = None
        self.all_backtest_trades = None
        self.backtest_trades = None
        self.sweep_symbol = None
        self.metrics_filter = "All"

        # Replay state — HTML file generated after each backtest
        self.replay_html_path = None

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
        self.var_timeframe = add_field(1, "Timeframe:", "combo", "M5", ["M1", "M5", "M10", "M15", "M30", "H1", "H4", "D1"])
        
        two_months_ago = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        self.var_start_date = add_field(2, "Start Date (YYYY-MM-DD):", "entry", two_months_ago)
        self.var_end_date = add_field(3, "End Date (YYYY-MM-DD):", "entry", datetime.now().strftime("%Y-%m-%d"))
        
        self.var_strategy = add_field(4, "Strategy Model:", "combo", "Noir", ["Noir", "BreakerBlock", "DoubleTop", "TripleTop", "MLPattern"])
        self.var_balance = add_field(5, "Initial Balance ($):", "entry", "1000.0")
        self.var_risk_amount = add_field(6, "Risk Value:", "entry", "25.0")
        self.var_risk_type = add_field(7, "Risk Metric Type:", "combo", "fixed", ["fixed", "percentage"])
        self.var_rr = add_field(8, "Risk-to-Reward (RR):", "entry", "5.0")
        self.var_direction = add_field(9, "Allowed Direction:", "combo", "Both", ["Both", "Buys Only", "Sells Only"])

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
        
        # Segmented Filter Control
        filter_frame = tk.Frame(self.metrics_frame, bg=self.colors["card_bg"])
        filter_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.metrics_filter = "All"
        self.filter_buttons = {}
        
        def on_filter_click(name):
            self.select_metrics_filter(name)
            
        for name in ["All", "Buys", "Sells"]:
            btn = tk.Button(
                filter_frame, text=name.upper(),
                command=lambda n=name: on_filter_click(n),
                bg="#2C2C2C", fg="#CCCCCC", activebackground="#00ADB5", activeforeground="#121212",
                bd=0, relief="flat", font=("Segoe UI", 7, "bold"),
                padx=8, pady=3, cursor="hand2"
            )
            btn.pack(side="left", expand=True, fill="x", padx=2)
            self.filter_buttons[name] = btn
            
        self.update_metrics_filter_ui()
        
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

    def select_metrics_filter(self, filter_name):
        self.metrics_filter = filter_name
        self.update_metrics_filter_ui()
        self.plot_equity_curve()

    def update_metrics_filter_ui(self):
        for name, btn in self.filter_buttons.items():
            if name == self.metrics_filter:
                btn.config(bg="#00ADB5", fg="#121212")
            else:
                btn.config(bg="#2C2C2C", fg="#CCCCCC")

    def plot_equity_curve(self):
        symbol = self.var_symbols.get().split(",")[0].strip()
        detailed_file = os.path.join(BACKTEST_SUMMARY_DIR, f"detailed_results_{symbol}.csv")
        
        if not os.path.exists(detailed_file):
            return
            
        try:
            df_results = pd.read_csv(detailed_file)
            if df_results.empty:
                return

            init_bal = float(self.var_balance.get().strip())
            risk_amount = float(self.var_risk_amount.get().strip())
            rr = float(self.var_rr.get().strip())

            # Apply UI metrics filter
            filter_val = self.metrics_filter
            if filter_val == "Buys":
                df_results = df_results[df_results["Stop_Loss"] < df_results["Entry"]].copy()
            elif filter_val == "Sells":
                df_results = df_results[df_results["Stop_Loss"] > df_results["Entry"]].copy()

            # Reconstruct balance curve chronologically
            balances = [init_bal]
            current_bal = init_bal
            for _, row in df_results.iterrows():
                res = row["Result"]
                risk_val = risk_amount
                if self.var_risk_type.get() == "percentage":
                    risk_val = (risk_amount / 100.0) * current_bal

                if res == "TP":
                    current_bal += risk_val * rr
                elif res == "SL":
                    current_bal -= risk_val
                balances.append(current_bal)
            
            balances = np.array(balances)
            
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
            self.all_backtest_trades = df_results[df_results["Result"].isin(["TP", "SL"])].copy()
            self.combo_filter.set("All")
            self.apply_gui_trade_filter()

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

    def apply_gui_trade_filter(self, event=None):
        if self.all_backtest_trades is None or self.all_backtest_trades.empty:
            return

        filter_val = self.combo_filter.get()
        if filter_val == "Buys":
            self.backtest_trades = self.all_backtest_trades[self.all_backtest_trades["Stop_Loss"] < self.all_backtest_trades["Entry"]].copy()
        elif filter_val == "Sells":
            self.backtest_trades = self.all_backtest_trades[self.all_backtest_trades["Stop_Loss"] > self.all_backtest_trades["Entry"]].copy()
        else:
            self.backtest_trades = self.all_backtest_trades.copy()

        trade_options = []
        for i, row in self.backtest_trades.reset_index(drop=True).iterrows():
            is_buy = row['Stop_Loss'] < row['Entry']
            t_type = "BUY" if is_buy else "SELL"
            trade_options.append(f"#{i+1}: {row['Occurrence']} | {t_type} {row['Result']} @ {row['Entry']:.5f}")

        self.combo_trades.config(values=trade_options)
        if trade_options:
            self.combo_trades.current(0)
            self.plot_selected_trade()
        else:
            for widget in self.pattern_chart_frame.winfo_children():
                widget.destroy()
            self.create_placeholder_pattern_chart()

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

            # Calculate Ask and Bid price dynamically based on spread and precision
            trig_row = df.iloc[trig_loc]
            spread_val = float(trig_row.get("spread", 0))
            
            sample_price = df["Close"].iloc[0]
            decimals = len(str(sample_price).split(".")[1]) if "." in str(sample_price) else 5
            point_size = 10 ** (-decimals)
            
            trig_bid = float(trig_row["Close"])
            trig_ask = trig_bid + (spread_val * point_size)

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
            ax0.axhline(trig_bid, color="#FF1744", linestyle="dotted", linewidth=1.2)
            ax0.axhline(trig_ask, color="#00B0FF", linestyle="dotted", linewidth=1.2)
            
            ax0.annotate(f"SL  {sl:.5f}",   xy=(1, sl),    xycoords=("axes fraction", "data"), color="#FF1744", fontsize=7, ha="right", va="bottom")
            ax0.annotate(f"Entry  {entry:.5f}", xy=(1, entry), xycoords=("axes fraction", "data"), color="#00B0FF", fontsize=7, ha="right", va="bottom")
            ax0.annotate(f"TP  {tp:.5f}",    xy=(1, tp),    xycoords=("axes fraction", "data"), color="#00E676", fontsize=7, ha="right", va="bottom")
            ax0.annotate(f"Bid  {trig_bid:.5f}", xy=(1, trig_bid), xycoords=("axes fraction", "data"), color="#FF1744", fontsize=7, ha="right", va="top")
            ax0.annotate(f"Ask  {trig_ask:.5f}", xy=(1, trig_ask), xycoords=("axes fraction", "data"), color="#00B0FF", fontsize=7, ha="right", va="bottom")

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
            "rr": rr_val,
            "direction": self.var_direction.get()
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
    # LIVE REPLAY  —  TradingView Lightweight Charts (browser-based)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_replay_tab(self):
        self.tab_replay.grid_columnconfigure(0, weight=1)
        self.tab_replay.grid_rowconfigure(0, weight=1)

        container = tk.Frame(self.tab_replay, bg="#0D0D0D")
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(container, bg="#0D0D0D")
        hdr.grid(row=0, column=0, sticky="ew", padx=30, pady=(30, 0))

        tk.Label(hdr, text="LIVE BACKTEST REPLAY",
                 bg="#0D0D0D", fg=self.colors["accent"],
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Powered by TradingView Lightweight Charts",
                 bg="#0D0D0D", fg=self.colors["text_muted"],
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

        # ── Status card ───────────────────────────────────────────────────────
        card = tk.Frame(container, bg="#1A1A1A",
                        highlightbackground="#2C2C2C", highlightthickness=1)
        card.grid(row=1, column=0, sticky="", padx=60, pady=30, ipadx=30, ipady=24)

        self.replay_status_icon = tk.Label(
            card, text="⏳", bg="#1A1A1A",
            font=("Segoe UI", 36), fg=self.colors["text_muted"]
        )
        self.replay_status_icon.pack(pady=(20, 8))

        self.replay_status_lbl = tk.Label(
            card, text="Run a backtest to enable Live Replay",
            bg="#1A1A1A", fg=self.colors["text_muted"],
            font=("Segoe UI", 11)
        )
        self.replay_status_lbl.pack()

        self.replay_detail_lbl = tk.Label(
            card, text="",
            bg="#1A1A1A", fg=self.colors["text_muted"],
            font=("Segoe UI", 9)
        )
        self.replay_detail_lbl.pack(pady=(4, 20))

        self.btn_launch_replay = ttk.Button(
            card, text="  🚀  LAUNCH REPLAY IN BROWSER  ",
            command=self._launch_replay, style="Action.TButton"
        )
        self.btn_launch_replay.pack(ipadx=16, ipady=6, pady=(0, 20))
        self.btn_launch_replay.config(state="disabled")

        # ── Feature grid ──────────────────────────────────────────────────────
        feat_frame = tk.Frame(container, bg="#0D0D0D")
        feat_frame.grid(row=2, column=0, sticky="ew", padx=60, pady=(0, 30))
        features = [
            ("🕯️", "Real Candlesticks", "Full OHLCV via Lightweight Charts"),
            ("▲▼", "Trade Markers", "TP/SL arrows on exact trigger candles"),
            ("─ ─", "Price Lines", "Entry · Stop Loss · Take Profit levels"),
            ("⚡", "Speed Control", "1× · 5× · 25× · 100× · ∞ jump-to-trade"),
            ("📊", "Live Metrics", "Balance · P&L · Win Rate update in real-time"),
            ("🔍", "Full Interactivity", "Zoom, pan, crosshair, price scale"),
        ]
        for col, (icon, title, desc) in enumerate(features):
            f = tk.Frame(feat_frame, bg="#141414",
                         highlightbackground="#222", highlightthickness=1)
            f.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            feat_frame.grid_columnconfigure(col, weight=1)
            tk.Label(f, text=icon, bg="#141414", font=("Segoe UI", 18)).pack(pady=(10, 2))
            tk.Label(f, text=title, bg="#141414", fg=self.colors["text"],
                     font=("Segoe UI", 8, "bold")).pack()
            tk.Label(f, text=desc, bg="#141414", fg=self.colors["text_muted"],
                     font=("Segoe UI", 7), wraplength=110).pack(pady=(2, 10))

    # ── Data preparation ──────────────────────────────────────────────────────

    def load_replay_data(self, symbol, timeframe_name):
        from currency.settings import HISTORY_DATA_DIR, BACKTEST_SUMMARY_DIR as BSD
        price_file  = os.path.join(HISTORY_DATA_DIR, f"{symbol}_data_{timeframe_name}.csv")
        trades_file = os.path.join(BSD, f"detailed_results_{symbol}.csv")

        if not os.path.exists(price_file) or not os.path.exists(trades_file):
            return

        self.replay_status_lbl.config(text="Preparing replay data…",
                                      fg=self.colors["accent"])
        self.replay_detail_lbl.config(text="")
        self.update_idletasks()

        try:
            html_path = self._generate_replay_html(symbol, timeframe_name,
                                                    price_file, trades_file)
            self.replay_html_path = html_path
            self.replay_status_icon.config(text="✅", fg=self.colors["success"])
            self.replay_status_lbl.config(
                text=f"Replay ready  ·  {symbol} ({timeframe_name})",
                fg=self.colors["success"]
            )
            # count trades
            import pandas as _pd
            tdf = _pd.read_csv(trades_file)
            n_trades = len(tdf[tdf["Result"].isin(["TP", "SL"])])
            import os as _os
            price_rows = sum(1 for _ in open(price_file)) - 1
            self.replay_detail_lbl.config(
                text=f"{price_rows:,} candles · {n_trades} closed trades",
                fg=self.colors["text_muted"]
            )
            self.btn_launch_replay.config(state="normal")
            self.log_message(
                f"[INFO] Live Replay ready for {symbol} — click LAUNCH REPLAY in the Live Replay tab.\n",
                "INFO"
            )
        except Exception as exc:
            self.replay_status_icon.config(text="❌", fg=self.colors["danger"])
            self.replay_status_lbl.config(text=f"Replay generation failed: {exc}",
                                          fg=self.colors["danger"])

    def _generate_replay_html(self, symbol, timeframe_name, price_file, trades_file):
        import pandas as pd
        import json as _json

        # ── Load & convert price data ──────────────────────────────────────────
        price_df = pd.read_csv(price_file)
        price_df["time"] = pd.to_datetime(price_df["time"])
        rename = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "tick_volume": "Volume"}
        price_df.rename(columns=rename, inplace=True)

        # Determine decimals dynamically to avoid hardcoding symbol points
        sample_price = price_df["Close"].iloc[0]
        decimals = len(str(sample_price).split(".")[1]) if "." in str(sample_price) else 5
        point_size = 10 ** (-decimals)

        candles = []
        for _, row in price_df.iterrows():
            ts = int(row["time"].timestamp())
            c_val = round(float(row["Close"]), 6)
            spread_val = float(row.get("spread", 0))
            ask_val = round(c_val + (spread_val * point_size), 6)
            candles.append({
                "time": ts,
                "open":  round(float(row["Open"]),  6),
                "high":  round(float(row["High"]),  6),
                "low":   round(float(row["Low"]),   6),
                "close": c_val,
                "ask":   ask_val,
            })

        # ── Load & convert trade data ──────────────────────────────────────────
        trades_df = pd.read_csv(trades_file)
        trades_df["Occurrence"] = pd.to_datetime(trades_df["Occurrence"])
        trades_df = (trades_df[trades_df["Result"].isin(["TP", "SL"])]
                     .sort_values("Occurrence").reset_index(drop=True))

        trades = []
        for _, row in trades_df.iterrows():
            ts = int(row["Occurrence"].timestamp())
            trades.append({
                "time":    ts,
                "entry":   round(float(row["Entry"]),     6),
                "sl":      round(float(row["Stop_Loss"]), 6),
                "tp":      round(float(row["Take_Profit"]), 6),
                "result":  str(row["Result"]),
                "balance": round(float(row["Balance"]),   2),
            })

        try:
            init_balance = float(self.var_balance.get())
        except (ValueError, AttributeError):
            init_balance = 1000.0

        candles_json = _json.dumps(candles, separators=(",", ":"))
        trades_json  = _json.dumps(trades,  separators=(",", ":"))

        # ── HTML template ─────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Replay · {symbol} ({timeframe_name})</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0D0D0D;color:#EEE;font-family:'Segoe UI',sans-serif;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
  #chart-wrap{{flex:1;min-height:0;position:relative}}
  #chart{{width:100%;height:100%}}
  #metrics{{display:flex;align-items:center;background:#111;padding:6px 20px;gap:28px;border-top:1px solid #1E1E1E;flex-shrink:0}}
  .met{{display:flex;flex-direction:column}}
  .met-lbl{{font-size:9px;color:#555;text-transform:uppercase;letter-spacing:.8px}}
  .met-val{{font-size:13px;font-weight:700;color:#CCC;margin-top:1px}}
  .pos{{color:#00E676}} .neg{{color:#FF1744}} .acc{{color:#00ADB5}}
  #controls{{display:flex;align-items:center;background:#141414;padding:7px 16px;gap:10px;border-top:1px solid #1E1E1E;flex-shrink:0}}
  button{{background:#1E1E1E;color:#CCC;border:1px solid #2C2C2C;padding:5px 14px;cursor:pointer;border-radius:3px;font-size:11px;font-weight:600;font-family:'Segoe UI',sans-serif;transition:background .12s,border-color .12s}}
  button:hover{{background:#2A2A2A;border-color:#00ADB5;color:#FFF}}
  button.play{{background:#00ADB5;color:#000;border-color:#00ADB5}}
  button.play:hover{{background:#00C8D4}}
  button.pause{{background:#FF9800;color:#000;border-color:#FF9800}}
  button.stop-btn{{background:#1A0000;border-color:#FF1744;color:#FF1744}}
  .sp-grp{{display:flex;gap:3px}}
  .sp{{padding:4px 9px;font-size:10px;min-width:32px}}
  .sp.on{{background:#00ADB5;color:#000;border-color:#00ADB5}}
  #prog-wrap{{flex:1;background:#1A1A1A;height:5px;border-radius:3px;cursor:pointer;position:relative}}
  #prog-wrap:hover{{background:#252525}}
  #prog{{background:#00ADB5;height:100%;border-radius:3px;width:0%;pointer-events:none;transition:width .04s linear}}
  #prog-lbl{{font-size:9px;color:#444;white-space:nowrap;min-width:110px;text-align:right}}
  #title-overlay{{position:absolute;top:10px;left:12px;font-size:11px;font-weight:700;color:#00ADB5;pointer-events:none;z-index:10;text-shadow:0 1px 4px #000}}
  .seek-marker{{position:absolute;top:-2px;width:4px;height:9px;border-radius:2px;cursor:pointer;transition:transform .1s,background .1s;z-index:5}}
  .seek-marker:hover{{transform:scale(1.5);background:#FFF !important;z-index:10}}
</style>
</head>
<body>
<div id="chart-wrap">
  <div id="chart"></div>
  <div id="title-overlay">{symbol} &nbsp;·&nbsp; {timeframe_name} &nbsp;·&nbsp; Live Backtest Replay</div>
</div>

<div id="metrics">
  <div class="met"><span class="met-lbl">Balance</span><span class="met-val" id="m-bal">—</span></div>
  <div class="met"><span class="met-lbl">Net P&amp;L</span><span class="met-val" id="m-pnl">—</span></div>
  <div class="met"><span class="met-lbl">Wins</span><span class="met-val pos" id="m-wins">0</span></div>
  <div class="met"><span class="met-lbl">Losses</span><span class="met-val neg" id="m-losses">0</span></div>
  <div class="met"><span class="met-lbl">Win Rate</span><span class="met-val acc" id="m-wr">0.0%</span></div>
  <div class="met"><span class="met-lbl">Drawdown</span><span class="met-val neg" id="m-dd">0.0%</span></div>
  <div class="met" style="margin-left:auto"><span class="met-lbl">Last Trade</span><span class="met-val" id="m-last">—</span></div>
</div>

<div id="controls">
  <button class="stop-btn" onclick="resetReplay()">◀◀&nbsp; RESET</button>
  <button id="btn-play" class="play" onclick="togglePlay()">▶&nbsp; PLAY</button>
  <span style="color:#444;font-size:11px;margin-left:6px">Filter</span>
  <select id="sel-filter" onchange="changeFilter(this.value)" style="background:#1E1E1E;color:#CCC;border:1px solid #2C2C2C;padding:4px 8px;font-size:11px;font-weight:600;font-family:'Segoe UI',sans-serif;border-radius:3px;cursor:pointer;outline:none">
    <option value="all">All Setups</option>
    <option value="buys">Buys Only</option>
    <option value="sells">Sells Only</option>
  </select>
  <span style="color:#444;font-size:11px;margin-left:6px">Jump to</span>
  <select id="sel-trade" onchange="jumpToTrade(this.value)" style="background:#1E1E1E;color:#CCC;border:1px solid #2C2C2C;padding:4px 8px;font-size:11px;font-weight:600;font-family:'Segoe UI',sans-serif;border-radius:3px;cursor:pointer;outline:none;max-width:150px">
    <option value="">-- Choose Setup --</option>
  </select>
  <span style="color:#444;font-size:11px;margin-left:6px">Speed</span>
  <div class="sp-grp">
    <button class="sp on"  onclick="setSpd(1,this)">1×</button>
    <button class="sp"     onclick="setSpd(5,this)">5×</button>
    <button class="sp"     onclick="setSpd(25,this)">25×</button>
    <button class="sp"     onclick="setSpd(100,this)">100×</button>
    <button class="sp"     onclick="setSpd(0,this)">∞</button>
  </div>
  <div id="prog-wrap" onclick="seekTo(event)"><div id="prog"></div></div>
  <span id="prog-lbl">0 / 0</span>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────────
const INIT_BAL   = {init_balance};
const ALL_C      = {candles_json};
const ALL_T      = {trades_json};

// ── Chart ─────────────────────────────────────────────────────────────────────
const chartEl = document.getElementById('chart');
const chart = LightweightCharts.createChart(chartEl, {{
  layout:         {{ background:{{type:'solid',color:'#0D0D0D'}}, textColor:'#666' }},
  grid:           {{ vertLines:{{color:'#141414'}}, horzLines:{{color:'#141414'}} }},
  rightPriceScale:{{ borderColor:'#1E1E1E' }},
  timeScale:      {{ borderColor:'#1E1E1E', timeVisible:true, secondsVisible:false, rightOffset:12 }},
  crosshair:      {{ mode:LightweightCharts.CrosshairMode.Normal }},
}});

const series = chart.addCandlestickSeries({{
  upColor:'#00E676', downColor:'#FF1744',
  borderUpColor:'#00E676', borderDownColor:'#FF1744',
  wickUpColor:'#00C853',   wickDownColor:'#D50000',
}});

new ResizeObserver(()=>chart.resize(chartEl.clientWidth, chartEl.clientHeight))
  .observe(chartEl);

// ── State ─────────────────────────────────────────────────────────────────────
let rIdx=0, playing=false, spd=1, tid=null;
let bal=INIT_BAL, peakBal=INIT_BAL, wins=0, losses=0, lastT=null;
let tradeSeriesList=[];
let filterType='all';
let activeTrades=ALL_T;
let bidLine = null, askLine = null;

// ── Formatting ────────────────────────────────────────────────────────────────
const fmt  = n=>n.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
const fmtP = n=>(n>=0?'+$':'-$')+fmt(Math.abs(n));
const fmtR = n=>n.toFixed(5);

// ── Pre-calculate Trade Lifecycles ───────────────────────────────────────────
function initTradeLifecycles() {{
  for (let t of ALL_T) {{
    t.sigIdx = ALL_C.findIndex(c => c.time === t.time);
    if (t.sigIdx === -1) continue;

    const isBuy = t.sl < t.entry;

    // Find entry index (when price first reaches entry price after signal)
    t.entryIdx = -1;
    for (let i = t.sigIdx + 1; i < ALL_C.length; i++) {{
      const c = ALL_C[i];
      if (isBuy) {{
        if (c.low <= t.entry) {{
          t.entryIdx = i;
          break;
        }}
      }} else {{
        if (c.high >= t.entry) {{
          t.entryIdx = i;
          break;
        }}
      }}
    }}

    // Find exit index (when price hits TP or SL after entry)
    t.exitIdx = -1;
    t.exitReason = null;
    if (t.entryIdx !== -1) {{
      for (let i = t.entryIdx; i < ALL_C.length; i++) {{
        const c = ALL_C[i];
        if (isBuy) {{
          const hitSL = c.low <= t.sl;
          const hitTP = c.high >= t.tp;
          if (hitSL && t.result === 'SL') {{
            t.exitIdx = i;
            t.exitReason = 'SL';
            break;
          }} else if (hitTP && t.result === 'TP') {{
            t.exitIdx = i;
            t.exitReason = 'TP';
            break;
          }} else if (hitSL || hitTP) {{
            t.exitIdx = i;
            t.exitReason = hitSL ? 'SL' : 'TP';
            break;
          }}
        }} else {{
          const hitSL = c.high >= t.sl;
          const hitTP = c.low <= t.tp;
          if (hitSL && t.result === 'SL') {{
            t.exitIdx = i;
            t.exitReason = 'SL';
            break;
          }} else if (hitTP && t.result === 'TP') {{
            t.exitIdx = i;
            t.exitReason = 'TP';
            break;
          }} else if (hitSL || hitTP) {{
            t.exitIdx = i;
            t.exitReason = hitSL ? 'SL' : 'TP';
            break;
          }}
        }}
      }}
    }}
  }}
}}

function drawSeekMarkers() {{
  const progWrap = document.getElementById('prog-wrap');
  document.querySelectorAll('.seek-marker').forEach(m => m.remove());

  for (let t of activeTrades) {{
    if (t.sigIdx === -1) continue;

    const idx = t.entryIdx !== -1 ? t.entryIdx : t.sigIdx;
    const pct = (idx / ALL_C.length) * 100;

    const marker = document.createElement('div');
    marker.className = 'seek-marker';
    marker.style.left = pct + '%';
    
    const isTP = t.result === 'TP';
    marker.style.background = t.entryIdx !== -1 ? (isTP ? '#00E676' : '#FF1744') : '#00B0FF';
    marker.title = (t.entryIdx !== -1 ? t.result : 'PENDING') + ' @ ' + fmtR(t.entry) + ' (Click to jump)';

    marker.addEventListener('click', e => {{
      e.stopPropagation();
      const target = t.entryIdx !== -1 ? Math.max(0, t.entryIdx - 10) : Math.max(0, t.sigIdx - 5);
      resetReplay();
      advance(target);
      chart.timeScale().scrollToRealTime();
    }});

    progWrap.appendChild(marker);
  }}
}}

function changeFilter(type) {{
  filterType = type;
  if (type === 'all') {{
    activeTrades = ALL_T;
  }} else if (type === 'buys') {{
    activeTrades = ALL_T.filter(t => t.sl < t.entry);
  }} else if (type === 'sells') {{
    activeTrades = ALL_T.filter(t => t.sl > t.entry);
  }}
  resetReplay();
  drawSeekMarkers();
  drawTradeDropdown();
}}

function drawTradeDropdown() {{
  const sel = document.getElementById('sel-trade');
  sel.innerHTML = '<option value="">-- Choose Setup --</option>';

  activeTrades.forEach((t, i) => {{
    if (t.sigIdx === -1) return;
    const isBuy = t.sl < t.entry;
    const typeStr = isBuy ? 'BUY' : 'SELL';
    const d = new Date(t.time * 1000);
    const dateStr = d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0') + ' ' + String(d.getHours()).padStart(2,'0') + ':' + String(d.getMinutes()).padStart(2,'0');
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = '#' + (i+1) + ': ' + typeStr + ' ' + t.result + ' @ ' + fmtR(t.entry) + ' (' + dateStr + ')';
    sel.appendChild(opt);
  }});
}}

function jumpToTrade(idxStr) {{
  if (idxStr === '') return;
  const idx = parseInt(idxStr);
  const t = activeTrades[idx];
  if (!t) return;
  const target = t.entryIdx !== -1 ? Math.max(0, t.entryIdx - 10) : Math.max(0, t.sigIdx - 5);
  resetReplay();
  advance(target);
  chart.timeScale().scrollToRealTime();
  document.getElementById('sel-trade').value = '';
}}

initTradeLifecycles();
drawSeekMarkers();
drawTradeDropdown();

// ── Metrics & State Update ────────────────────────────────────────────────────
function updStateAndMetrics() {{
  bal = INIT_BAL;
  peakBal = INIT_BAL;
  wins = 0;
  losses = 0;
  lastT = null;

  for (let t of activeTrades) {{
    if (t.sigIdx === -1) continue;

    if (t.exitIdx !== -1 && rIdx >= t.exitIdx) {{
      bal = t.balance;
      if (bal > peakBal) peakBal = bal;
      if (t.result === 'TP') wins++; else losses++;
      lastT = t;
    }}
  }}

  const pnl = bal - INIT_BAL;
  const tot = wins + losses;
  const wr = tot > 0 ? (wins / tot * 100).toFixed(1) : '0.0';
  const dd = peakBal > 0 ? ((peakBal - bal) / peakBal * 100).toFixed(1) : '0.0';

  document.getElementById('m-bal').textContent  = '$' + fmt(bal);
  const pe = document.getElementById('m-pnl');
  pe.textContent = fmtP(pnl);
  pe.className = 'met-val ' + (pnl >= 0 ? 'pos' : 'neg');
  document.getElementById('m-wins').textContent   = wins;
  document.getElementById('m-losses').textContent = losses;
  document.getElementById('m-wr').textContent     = wr + '%';
  document.getElementById('m-dd').textContent     = dd + '%';

  if (lastT) {{
    const le = document.getElementById('m-last');
    le.textContent = lastT.result + ' @ ' + fmtR(lastT.entry);
    le.className = 'met-val ' + (lastT.result === 'TP' ? 'pos' : 'neg');
  }} else {{
    const le = document.getElementById('m-last');
    le.textContent = '—';
    le.className = 'met-val';
  }}

  const pct = ALL_C.length > 0 ? rIdx / ALL_C.length * 100 : 0;
  document.getElementById('prog').style.width = pct + '%';
  document.getElementById('prog-lbl').textContent = rIdx.toLocaleString() + ' / ' + ALL_C.length.toLocaleString();
}}

function updMetrics() {{
  updStateAndMetrics();
}}

// ── Draw Trade Setup Levels (Entry, TP, SL) ──────────────────────────────────
function updTradeSeries() {{
  for (let t of ALL_T) {{
    if (t.sigIdx === -1) continue;

    const isActive = activeTrades.includes(t);
    if (rIdx < t.sigIdx || !isActive) {{
      if (t.seriesCreated) {{
        chart.removeSeries(t.entrySeries);
        chart.removeSeries(t.tpSeries);
        chart.removeSeries(t.slSeries);
        t.seriesCreated = false;
      }}
      continue;
    }}

    if (!t.seriesCreated) {{
      t.entrySeries = chart.addLineSeries({{
        color: '#00B0FF',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        priceLineVisible: false,
        lastPriceAnimationMode: LightweightCharts.LastPriceAnimationMode.Disabled,
        crosshairMarkerVisible: false
      }});
      t.tpSeries = chart.addLineSeries({{
        color: '#00E676',
        lineWidth: 1.5,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false,
        lastPriceAnimationMode: LightweightCharts.LastPriceAnimationMode.Disabled,
        crosshairMarkerVisible: false
      }});
      t.slSeries = chart.addLineSeries({{
        color: '#FF1744',
        lineWidth: 1.5,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false,
        lastPriceAnimationMode: LightweightCharts.LastPriceAnimationMode.Disabled,
        crosshairMarkerVisible: false
      }});
      t.seriesCreated = true;
      tradeSeriesList.push(t);
    }}

    const start = t.sigIdx;
    let end = rIdx;

    if (t.exitIdx !== -1 && rIdx >= t.exitIdx) {{
      end = t.exitIdx;
    }}

    const entryData = [];
    const tpData = [];
    const slData = [];

    for (let i = start; i <= end; i++) {{
      const time = ALL_C[i].time;
      entryData.push({{ time: time, value: t.entry }});
      tpData.push({{ time: time, value: t.tp }});
      slData.push({{ time: time, value: t.sl }});
    }}

    t.entrySeries.setData(entryData);
    t.tpSeries.setData(tpData);
    t.slSeries.setData(slData);
  }}
}}

// ── Draw Trade Execution Markers ─────────────────────────────────────────────
function updMarkers() {{
  const activeMarkers = [];
  for (let t of activeTrades) {{
    if (t.sigIdx === -1) continue;

    if (rIdx >= t.sigIdx) {{
      const isBuy = t.sl < t.entry;
      activeMarkers.push({{
        time: t.time,
        position: isBuy ? 'belowBar' : 'aboveBar',
        color: '#00B0FF',
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        text: (isBuy ? 'BUY' : 'SELL') + ' LIMIT @ ' + fmtR(t.entry),
      }});
    }}
    if (t.entryIdx !== -1 && rIdx >= t.entryIdx) {{
      const isBuy = t.sl < t.entry;
      activeMarkers.push({{
        time: ALL_C[t.entryIdx].time,
        position: isBuy ? 'belowBar' : 'aboveBar',
        color: '#FFD600',
        shape: 'circle',
        text: 'FILLED @ ' + fmtR(t.entry),
      }});
    }}
    if (t.exitIdx !== -1 && rIdx >= t.exitIdx) {{
      const isTP = t.exitReason === 'TP';
      activeMarkers.push({{
        time: ALL_C[t.exitIdx].time,
        position: isTP ? 'aboveBar' : 'belowBar',
        color: isTP ? '#00E676' : '#FF1744',
        shape: isTP ? 'arrowUp' : 'arrowDown',
        text: t.exitReason + ' @ ' + fmtR(isTP ? t.tp : t.sl),
      }});
    }}
  }}
  activeMarkers.sort((a, b) => a.time - b.time);
  series.setMarkers(activeMarkers);
}}

// ── Advance Candles ───────────────────────────────────────────────────────────
function advance(n) {{
  rIdx = Math.min(rIdx + n, ALL_C.length);
  series.setData(ALL_C.slice(0, rIdx));
  
  if (bidLine) {{
    series.removePriceLine(bidLine);
    bidLine = null;
  }}
  if (askLine) {{
    series.removePriceLine(askLine);
    askLine = null;
  }}
  
  if (rIdx > 0) {{
    const currentCandle = ALL_C[rIdx - 1];
    
    bidLine = series.createPriceLine({{
      price: currentCandle.close,
      color: '#FF1744',
      lineWidth: 1.5,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: 'BID',
    }});
    
    askLine = series.createPriceLine({{
      price: currentCandle.ask,
      color: '#00B0FF',
      lineWidth: 1.5,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: 'ASK',
    }});
  }}

  updTradeSeries();
  updMarkers();
  updStateAndMetrics();
}}

// ── Tick loop ─────────────────────────────────────────────────────────────────
function tick() {{
  if (!playing) return;
  if (rIdx >= ALL_C.length) {{
    playing = false;
    const b = document.getElementById('btn-play');
    b.textContent = '▶  PLAY'; b.className = 'play';
    updStateAndMetrics(); return;
  }}
  if (spd === 0) {{
    advance(ALL_C.length - rIdx);
    chart.timeScale().scrollToRealTime();
    playing = false;
    const b = document.getElementById('btn-play');
    b.textContent = '▶  PLAY'; b.className = 'play';
    return;
  }} else {{
    advance(spd);
    chart.timeScale().scrollToRealTime();
    const dl = spd <= 1 ? 80 : spd <= 5 ? 28 : spd <= 25 ? 10 : 3;
    tid = setTimeout(tick, dl);
  }}
}}

// ── Controls ──────────────────────────────────────────────────────────────────
function togglePlay() {{
  playing = !playing;
  const b = document.getElementById('btn-play');
  if (playing) {{ b.textContent = '⏸  PAUSE'; b.className = 'pause'; tick(); }}
  else {{ b.textContent = '▶  PLAY'; b.className = 'play'; clearTimeout(tid); }}
}}

function resetReplay() {{
  playing = false; clearTimeout(tid);
  rIdx = 0;
  tradeSeriesList.forEach(s => {{
    chart.removeSeries(s.entrySeries);
    chart.removeSeries(s.tpSeries);
    chart.removeSeries(s.slSeries);
  }});
  tradeSeriesList = [];
  for (let t of ALL_T) {{
    t.seriesCreated = false;
    t.entrySeries = null;
    t.tpSeries = null;
    t.slSeries = null;
  }}
  if (bidLine) {{
    series.removePriceLine(bidLine);
    bidLine = null;
  }}
  if (askLine) {{
    series.removePriceLine(askLine);
    askLine = null;
  }}
  series.setData([]);
  series.setMarkers([]);
  document.getElementById('btn-play').textContent = '▶  PLAY';
  document.getElementById('btn-play').className = 'play';
  updStateAndMetrics();
}}

function setSpd(s,el){{
  spd=s;
  document.querySelectorAll('.sp').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
}}

function seekTo(e){{
  const r=e.currentTarget.getBoundingClientRect();
  const pct=(e.clientX-r.left)/r.width;
  const tgt=Math.floor(pct*ALL_C.length);
  if(tgt<=rIdx)return;
  advance(tgt-rIdx);
  updMetrics();
  chart.timeScale().scrollToRealTime();
}}

// ── Keyboard Navigation ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  if (e.key === ' ') {{
    e.preventDefault();
    togglePlay();
  }} else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {{
    e.preventDefault();
    if (!playing && rIdx < ALL_C.length) {{
      advance(1);
      updMetrics();
      chart.timeScale().scrollToRealTime();
    }}
  }} else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
    e.preventDefault();
    if (!playing && rIdx > 0) {{
      const target = rIdx - 1;
      resetReplay();
      advance(target);
      updMetrics();
      chart.timeScale().scrollToRealTime();
    }}
  }} else if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') {{
    e.preventDefault();
    const speeds = [1, 5, 25, 100, 0];
    let nextIdx = (speeds.indexOf(spd) + 1) % speeds.length;
    const nextSpd = speeds[nextIdx];
    const btn = document.querySelectorAll('.sp')[nextIdx];
    setSpd(nextSpd, btn);
  }} else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') {{
    e.preventDefault();
    const speeds = [1, 5, 25, 100, 0];
    let prevIdx = (speeds.indexOf(spd) - 1 + speeds.length) % speeds.length;
    const prevSpd = speeds[prevIdx];
    const btn = document.querySelectorAll('.sp')[prevIdx];
    setSpd(prevSpd, btn);
  }} else if (e.key === 'Escape' || e.key === 'r' || e.key === 'R') {{
    e.preventDefault();
    resetReplay();
  }}
}});

// ── Init ──────────────────────────────────────────────────────────────────────
updMetrics();
</script>
</body>
</html>"""

        out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "backtest_summary")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"live_replay_{symbol}_{timeframe_name}.html")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return out_path

    def _launch_replay(self):
        if not self.replay_html_path or not os.path.exists(self.replay_html_path):
            self.log_message("[WARN] No replay file found — run a backtest first.\n", "WARN")
            return
        webbrowser.open(f"file:///{self.replay_html_path.replace(os.sep, '/')}")

    # ─────────────────────────────────────────────────────────────────────────
    # ML TRAINING TAB
    # ─────────────────────────────────────────────────────────────────────────

    ML_SYMBOLS = [
        "Volatility 25 Index", "Volatility 10 Index", "Volatility 15 Index",
        "Volatility 100 Index", "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD",
    ]

    def _build_ml_train_tab(self):
        self.tab_ml_train.grid_columnconfigure(0, weight=1)
        self.tab_ml_train.grid_rowconfigure(2, weight=1)

        # ── Row 0: Parameters card ──────────────────────────────────────────
        params_card = ttk.Frame(self.tab_ml_train, style="Card.TFrame")
        params_card.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        params_card.columnconfigure(1, weight=1)

        lbl = tk.Label(params_card, text="TRAINING PARAMETERS",
                       bg=self.colors["card_bg"], fg=self.colors["accent"],
                       font=("Segoe UI", 10, "bold"))
        lbl.grid(row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(8, 4))

        row = 1
        tk.Label(params_card, text="Symbol:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=row, column=0, sticky="w", padx=12, pady=3)
        self.ml_symbol = ttk.Combobox(params_card, values=self.ML_SYMBOLS, state="readonly", width=22)
        self.ml_symbol.set("Volatility 25 Index")
        self.ml_symbol.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=3)

        tk.Label(params_card, text="Timeframe:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=row, column=2, sticky="w", padx=8, pady=3)
        self.ml_tf = ttk.Combobox(params_card, values=["M1","M5","M10","M15","M30","H1","H4","D1"],
                                  state="readonly", width=6)
        self.ml_tf.set("M10")
        self.ml_tf.grid(row=row, column=3, sticky="ew", padx=(0, 12), pady=3)

        row += 1
        tk.Label(params_card, text="Start Date:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=row, column=0, sticky="w", padx=12, pady=3)
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        self.ml_start = tk.Entry(params_card, bg="#2A2A2A", fg=self.colors["text"],
                                 insertbackground=self.colors["text"], bd=0, relief="flat",
                                 highlightthickness=1, highlightbackground="#3A3A3A",
                                 highlightcolor=self.colors["accent"], font=("Segoe UI", 9))
        self.ml_start.insert(0, three_months_ago)
        self.ml_start.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=3)

        tk.Label(params_card, text="End Date:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=row, column=2, sticky="w", padx=8, pady=3)
        self.ml_end = tk.Entry(params_card, bg="#2A2A2A", fg=self.colors["text"],
                               insertbackground=self.colors["text"], bd=0, relief="flat",
                               highlightthickness=1, highlightbackground="#3A3A3A",
                               highlightcolor=self.colors["accent"], font=("Segoe UI", 9))
        self.ml_end.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.ml_end.grid(row=row, column=3, sticky="ew", padx=(0, 12), pady=3)

        row += 1
        tk.Label(params_card, text="RR:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=row, column=0, sticky="w", padx=12, pady=3)
        self.ml_rr = tk.Entry(params_card, bg="#2A2A2A", fg=self.colors["text"],
                              insertbackground=self.colors["text"], bd=0, relief="flat",
                              highlightthickness=1, highlightbackground="#3A3A3A",
                              highlightcolor=self.colors["accent"], font=("Segoe UI", 9), width=8)
        self.ml_rr.insert(0, "5.0")
        self.ml_rr.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=3)

        tk.Label(params_card, text="Test Size:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=row, column=2, sticky="w", padx=8, pady=3)
        self.ml_test_size = tk.Entry(params_card, bg="#2A2A2A", fg=self.colors["text"],
                                     insertbackground=self.colors["text"], bd=0, relief="flat",
                                     highlightthickness=1, highlightbackground="#3A3A3A",
                                     highlightcolor=self.colors["accent"], font=("Segoe UI", 9), width=8)
        self.ml_test_size.insert(0, "0.2")
        self.ml_test_size.grid(row=row, column=3, sticky="w", padx=(0, 12), pady=3)

        row += 1
        btn_frame = tk.Frame(params_card, bg=self.colors["card_bg"])
        btn_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=12, pady=(4, 10))

        for label, delta in [("Last 3 Months", 90), ("Last 6 Months", 180), ("Last Year", 365)]:
            b = tk.Button(btn_frame, text=label,
                          command=lambda d=delta: self._ml_set_date_range(d),
                          bg="#2C2C2C", fg="#CCCCCC", activebackground=self.colors["accent"],
                          activeforeground="#121212", bd=0, relief="flat",
                          font=("Segoe UI", 8, "bold"), padx=10, pady=3, cursor="hand2")
            b.pack(side="left", padx=(0, 8))

        self.ml_train_btn = ttk.Button(btn_frame, text="TRAIN MODEL",
                                       command=self.start_ml_training, style="Action.TButton")
        self.ml_train_btn.pack(side="right")

        # ── Row 1: Train log ────────────────────────────────────────────────
        log_card = ttk.Frame(self.tab_ml_train, style="Card.TFrame")
        log_card.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        log_card.grid_columnconfigure(0, weight=1)

        tk.Label(log_card, text="TRAINING LOG", bg=self.colors["card_bg"],
                 fg=self.colors["accent"], font=("Segoe UI", 10, "bold"))\
            .pack(anchor="w", padx=12, pady=(8, 4))

        self.ml_log = scrolledtext.ScrolledText(
            log_card, wrap=tk.WORD, bg="#0B0B0B", fg="#00FF66",
            insertbackground="#EEEEEE", font=("Consolas", 9), bd=0,
            highlightthickness=0, height=10
        )
        self.ml_log.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # ── Row 2: Saved models ─────────────────────────────────────────────
        models_card = ttk.Frame(self.tab_ml_train, style="Card.TFrame")
        models_card.grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 10))
        models_card.grid_columnconfigure(0, weight=1)
        models_card.grid_rowconfigure(1, weight=1)

        hdr_frame = tk.Frame(models_card, bg=self.colors["card_bg"])
        hdr_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        hdr_frame.columnconfigure(1, weight=1)

        tk.Label(hdr_frame, text="SAVED MODELS", bg=self.colors["card_bg"],
                 fg=self.colors["accent"], font=("Segoe UI", 10, "bold"))\
            .grid(row=0, column=0, sticky="w")

        self.ml_refresh_btn = tk.Button(hdr_frame, text="REFRESH",
                                        command=self.refresh_saved_models,
                                        bg="#2C2C2C", fg="#CCCCCC", activebackground=self.colors["accent"],
                                        activeforeground="#121212", bd=0, relief="flat",
                                        font=("Segoe UI", 8, "bold"), padx=8, pady=2, cursor="hand2")
        self.ml_refresh_btn.grid(row=0, column=1, sticky="e")

        list_frame = tk.Frame(models_card, bg=self.colors["card_bg"])
        list_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)

        self.ml_models_listbox = tk.Listbox(
            list_frame, bg="#1E1E1E", fg=self.colors["text"],
            selectbackground=self.colors["accent"], selectforeground="#FFFFFF",
            activestyle="none", font=("Consolas", 9), bd=0, highlightthickness=0, relief="flat"
        )
        self.ml_models_listbox.grid(row=0, column=0, sticky="nsew")

        scroll_m = tk.Scrollbar(list_frame, orient="vertical", command=self.ml_models_listbox.yview)
        scroll_m.grid(row=0, column=1, sticky="ns")
        self.ml_models_listbox.config(yscrollcommand=scroll_m.set)

        apply_frame = tk.Frame(models_card, bg=self.colors["card_bg"])
        apply_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 10))
        apply_frame.columnconfigure(1, weight=1)

        tk.Label(apply_frame, text="Threshold:", bg=self.colors["card_bg"],
                 fg=self.colors["text"], font=("Segoe UI", 9))\
            .grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.ml_threshold = tk.Entry(apply_frame, bg="#2A2A2A", fg=self.colors["text"],
                                     insertbackground=self.colors["text"], bd=0, relief="flat",
                                     highlightthickness=1, highlightbackground="#3A3A3A",
                                     highlightcolor=self.colors["accent"], font=("Segoe UI", 9), width=8)
        self.ml_threshold.insert(0, "0.58")
        self.ml_threshold.grid(row=0, column=1, sticky="w", padx=(0, 12))

        self.ml_apply_btn = ttk.Button(apply_frame, text="APPLY CONFIG TO SYMBOL",
                                       command=self.apply_model_config, style="Action.TButton")
        self.ml_apply_btn.grid(row=0, column=2, sticky="e")

        self.ml_status_lbl = tk.Label(apply_frame, text="", bg=self.colors["card_bg"],
                                      fg=self.colors["success"], font=("Segoe UI", 9))
        self.ml_status_lbl.grid(row=0, column=3, sticky="w", padx=(10, 0))

        # Populate saved models on init
        self.after(500, self.refresh_saved_models)

    def _ml_set_date_range(self, days_back):
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        self.ml_start.delete(0, tk.END)
        self.ml_start.insert(0, start)
        self.ml_end.delete(0, tk.END)
        self.ml_end.insert(0, end)

    def start_ml_training(self):
        self.ml_train_btn.config(state="disabled")
        self.ml_log.delete("1.0", tk.END)
        self.ml_log.insert(tk.END, "[INFO] Starting ML training...\n")
        self.ml_log.see(tk.END)

        self.ml_stream = StringIO()
        self.ml_thread = threading.Thread(target=self._run_ml_training, daemon=True)
        self.ml_thread.start()
        self.after(100, self._update_ml_log)

    def _run_ml_training(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = self.ml_stream
            sys.stderr = self.ml_stream

            symbol = self.ml_symbol.get()
            tf_name = self.ml_tf.get()
            start_str = self.ml_start.get().strip()
            end_str = self.ml_end.get().strip()
            rr = float(self.ml_rr.get().strip())
            test_size = float(self.ml_test_size.get().strip())

            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d")

            from currency.unified_trading import (
                initialize_mt5, shutdown_mt5, get_historical_data,
                prep_data, clean_data, detect_pivot_points
            )
            from currency.modules.ml_pattern import build_and_train_model

            tf_map = {
                "M1": 1, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440,
            }
            import MetaTrader5 as mt5
            mt5_tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M10": mt5.TIMEFRAME_M10,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
            }

            print(f"[INFO] Initializing MT5...")
            if not initialize_mt5():
                print("[ERROR] MT5 initialization failed.")
                return

            print(f"[INFO] Fetching {symbol} {tf_name} data from {start_str} to {end_str}...")
            ok = get_historical_data(symbol, mt5_tf_map[tf_name], tf_name, start_dt, end_dt)
            if not ok:
                print(f"[ERROR] Failed to fetch historical data for {symbol}.")
                shutdown_mt5()
                return

            df = prep_data(symbol, tf_name)
            print(f"[INFO] Loaded {len(df)} candles.")
            clean_data(df, symbol)
            detect_pivot_points(df, symbol)

            pivot_count = df["Is_High"].notna().sum() + df["Is_Low"].notna().sum()
            print(f"[INFO] Detected {pivot_count} pivot points.")

            result = build_and_train_model(df, symbol, RR=rr, test_size=test_size)
            if result.get("success"):
                print(f"\n[DONE] Model trained and saved for {symbol}.")
                print(f"       Test accuracy: {result['test_metrics']['accuracy']:.4f}")
                print(f"       Test precision: {result['test_metrics']['precision']:.4f}")
                print(f"       Test F1: {result['test_metrics']['f1']:.4f}")
                print(f"       Cutoff date: {result['cutoff']}")
            else:
                print(f"\n[ERROR] Training failed: {result.get('reason', 'unknown')}")

            shutdown_mt5()

        except Exception as e:
            print(f"\n[ERROR] ML training exception: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _update_ml_log(self):
        contents = self.ml_stream.getvalue()
        if contents:
            self.ml_log.insert(tk.END, contents)
            self.ml_log.see(tk.END)
            self.ml_stream.seek(0)
            self.ml_stream.truncate(0)

        if hasattr(self, 'ml_thread') and self.ml_thread and self.ml_thread.is_alive():
            self.after(100, self._update_ml_log)
        else:
            self.ml_train_btn.config(state="normal")
            self.refresh_saved_models()
            self.ml_log.insert(tk.END, "\n[INFO] Training finished.\n")
            self.ml_log.see(tk.END)

    def refresh_saved_models(self):
        from currency.modules.ml_pattern import list_saved_models
        self.ml_models_listbox.delete(0, tk.END)
        self._ml_models_data = []  # map listbox index → symbol name
        models = list_saved_models()
        if not models:
            self.ml_models_listbox.insert(tk.END, "  No saved models found.")
            self.ml_models_listbox.itemconfig(0, fg=self.colors["text_muted"])
            return
        for m in sorted(models, key=lambda x: x["symbol"]):
            self._ml_models_data.append(m["symbol"])
            meta = m["meta"]
            if meta:
                cutoff = meta.get("train_cutoff", "?")
                n_train = meta.get("n_train", "?")
                n_test = meta.get("n_test", "?")
                line = f"  {m['symbol']:<28} cutoff={cutoff}  train={n_train}  test={n_test}"
            else:
                line = f"  {m['symbol']:<28}  (no metadata)"
            self.ml_models_listbox.insert(tk.END, line)

    def apply_model_config(self):
        sel = self.ml_models_listbox.curselection()
        if not sel:
            self.ml_status_lbl.config(text="Select a model first.", fg=self.colors["warning"])
            return
        idx = sel[0]
        if not hasattr(self, '_ml_models_data') or idx >= len(self._ml_models_data):
            self.ml_status_lbl.config(text="Invalid selection.", fg=self.colors["danger"])
            return
        symbol = self._ml_models_data[idx]
        try:
            threshold = float(self.ml_threshold.get().strip())
            rr = float(self.ml_rr.get().strip())
        except ValueError:
            self.ml_status_lbl.config(text="Invalid RR or threshold.", fg=self.colors["danger"])
            return

        from currency.find_best_pattern import save_sweep_result
        ok, msg = save_sweep_result(symbol, {"RR": rr, "Threshold": threshold})
        if ok:
            self.ml_status_lbl.config(
                text=f"✓ Applied RR={rr:.1f} Thr={threshold:.2f} to {symbol}",
                fg=self.colors["success"]
            )
            self.log_message(f"[INFO] ML config applied: {symbol} — RR={rr:.1f} Threshold={threshold:.2f}\n", "INFO")
        else:
            self.ml_status_lbl.config(text=f"✗ {msg}", fg=self.colors["danger"])


if __name__ == "__main__":
    app = TradingBotGUI()
    app.mainloop()
