import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
from io import StringIO
from datetime import datetime, timedelta
from currency.unified_trading import main as run_trading_logic
from currency.modules import trading_pairs

class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Echelnet Unified Algorithmic Trading Terminal")
        self.geometry("1100x700")
        self.configure(bg="#121212")

        # Custom Styling & Colors
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

        # Apply styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure general styles
        self.style.configure(".", background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("Card.TFrame", background=self.colors["card_bg"], relief="flat", borderwidth=1)
        
        # Labels
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=self.colors["accent"])
        self.style.configure("CardHeader.TLabel", background=self.colors["card_bg"], font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"])
        self.style.configure("CardLabel.TLabel", background=self.colors["card_bg"], font=("Segoe UI", 9), foreground=self.colors["text"])

        # Custom Buttons
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), foreground="#FFFFFF", background=self.colors["accent"], borderwidth=0, padding=8)
        self.style.map("Action.TButton",
            background=[("active", self.colors["accent_hover"]), ("disabled", "#444444")],
            foreground=[("disabled", "#888888")]
        )
        
        self.style.configure("Stop.TButton", font=("Segoe UI", 10, "bold"), foreground="#FFFFFF", background=self.colors["danger"], borderwidth=0, padding=8)
        self.style.map("Stop.TButton",
            background=[("active", "#FF5252"), ("disabled", "#444444")],
            foreground=[("disabled", "#888888")]
        )

        # Radio & Combo
        self.style.configure("TRadiobutton", background=self.colors["card_bg"], foreground=self.colors["text"], font=("Segoe UI", 9))
        self.style.configure("TCombobox", fieldbackground="#2A2A2A", background="#2A2A2A", foreground=self.colors["text"], arrowcolor=self.colors["accent"])

        # Main Layout
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Left Panel: Configurations & Controls
        self.left_panel = ttk.Frame(self, style="TFrame")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Title Block
        self.title_label = ttk.Label(self.left_panel, text="ECHELNET TRADING TERMINAL", style="Header.TLabel")
        self.title_label.pack(anchor="w", pady=(0, 15))

        # Card 1: Configuration Form
        self.config_card = ttk.Frame(self.left_panel, style="Card.TFrame")
        self.config_card.pack(fill="both", expand=True, pady=(0, 15))
        
        # Build Config Form UI
        self.create_config_form()

        # Card 2: Control Action Block
        self.control_card = ttk.Frame(self.left_panel, style="Card.TFrame")
        self.control_card.pack(fill="x", pady=(0, 0))
        self.create_control_block()

        # Right Panel: Live Terminal Output Logs
        self.right_panel = ttk.Frame(self, style="TFrame")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 15), pady=15)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)

        self.log_header = ttk.Label(self.right_panel, text="LIVE TERMINAL OUTPUT", style="Header.TLabel")
        self.log_header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Terminal Console Style Box
        self.log_area = scrolledtext.ScrolledText(
            self.right_panel, 
            wrap=tk.WORD, 
            bg="#0B0B0B", 
            fg="#00FF66", 
            insertbackground="#EEEEEE",
            font=("Consolas", 10),
            bd=1,
            highlightthickness=1,
            highlightbackground="#2C2C2C",
            highlightcolor=self.colors["accent"]
        )
        self.log_area.grid(row=1, column=0, sticky="nsew")

        # Color Tags for Logs
        self.log_area.tag_config("TP", foreground=self.colors["success"])
        self.log_area.tag_config("SL", foreground=self.colors["danger"])
        self.log_area.tag_config("ERROR", foreground=self.colors["danger"], font=("Consolas", 10, "bold"))
        self.log_area.tag_config("INFO", foreground=self.colors["accent"])
        self.log_area.tag_config("WARN", foreground=self.colors["warning"])

        # Initializing core tracking variables
        self.stop_event = threading.Event()
        self.bot_thread = None

    def create_config_form(self):
        # Card Header
        lbl = tk.Label(self.config_card, text="STRATEGY CONFIGURATIONS", bg=self.colors["card_bg"], fg=self.colors["accent"], font=("Segoe UI", 11, "bold"))
        lbl.pack(anchor="w", padx=15, pady=(15, 10))

        form_frame = tk.Frame(self.config_card, bg=self.colors["card_bg"])
        form_frame.pack(fill="both", expand=True, padx=15, pady=5)
        form_frame.columnconfigure(1, weight=1)

        # Field Helper Function
        def add_field(row, label_text, var_type="entry", default_val="", options=None):
            lbl_w = tk.Label(form_frame, text=label_text, bg=self.colors["card_bg"], fg=self.colors["text"], font=("Segoe UI", 9))
            lbl_w.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
            
            if var_type == "entry":
                var = tk.StringVar(value=default_val)
                entry = tk.Entry(form_frame, textvariable=var, bg="#2A2A2A", fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, relief="flat", highlightthickness=1, highlightbackground="#3A3A3A", highlightcolor=self.colors["accent"], font=("Segoe UI", 9))
                entry.grid(row=row, column=1, sticky="ew", pady=6)
                return var
            elif var_type == "combo":
                var = tk.StringVar(value=default_val)
                combo = ttk.Combobox(form_frame, textvariable=var, values=options, state="readonly")
                combo.grid(row=row, column=1, sticky="ew", pady=6)
                return var

        # Config fields
        default_pairs = ", ".join(trading_pairs.symbols)
        self.var_symbols = add_field(0, "Symbols (comma list):", "entry", default_pairs)
        self.var_timeframe = add_field(1, "Timeframe:", "combo", "M5", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        
        # Sensible Date Range Picker (Defaulting to recent 2 months to prevent MT5 "Invalid params" / size overload error)
        two_months_ago = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        self.var_start_date = add_field(2, "Start Date (YYYY-MM-DD):", "entry", two_months_ago)
        self.var_end_date = add_field(3, "End Date (YYYY-MM-DD):", "entry", datetime.now().strftime("%Y-%m-%d"))
        
        self.var_strategy = add_field(4, "Strategy Model:", "combo", "Noir", ["Noir", "BreakerBlock", "DoubleTop", "TripleTop"])
        self.var_balance = add_field(5, "Initial Balance ($):", "entry", "1000.0")
        self.var_risk_amount = add_field(6, "Risk Value:", "entry", "25.0")
        self.var_risk_type = add_field(7, "Risk Metric Type:", "combo", "fixed", ["fixed", "percentage"])
        self.var_rr = add_field(8, "Risk-to-Reward (RR):", "entry", "5.0")

    def create_control_block(self):
        lbl = tk.Label(self.control_card, text="OPERATIONAL CONTROLS", bg=self.colors["card_bg"], fg=self.colors["accent"], font=("Segoe UI", 11, "bold"))
        lbl.pack(anchor="w", padx=15, pady=(15, 10))

        btn_frame = tk.Frame(self.control_card, bg=self.colors["card_bg"])
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # Mode Selection
        self.mode_var = tk.StringVar(value="backtest")
        
        rb_backtest = ttk.Radiobutton(btn_frame, text="Run Backtest", variable=self.mode_var, value="backtest")
        rb_backtest.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        rb_live = ttk.Radiobutton(btn_frame, text="Execute Live Market Trading", variable=self.mode_var, value="live")
        rb_live.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # Start / Stop Buttons
        self.start_button = ttk.Button(btn_frame, text="START BOT ENGINE", command=self.start_bot, style="Action.TButton")
        self.start_button.grid(row=2, column=0, sticky="ew", padx=(0, 5))

        self.stop_button = ttk.Button(btn_frame, text="FORCE SHUTDOWN", command=self.stop_bot, style="Stop.TButton", state="disabled")
        self.stop_button.grid(row=2, column=1, sticky="ew", padx=(5, 0))

    def start_bot(self):
        # Validate Inputs
        try:
            start_dt = datetime.strptime(self.var_start_date.get().strip(), "%Y-%m-%d")
            end_dt = datetime.strptime(self.var_end_date.get().strip(), "%Y-%m-%d")
            init_bal = float(self.var_balance.get().strip())
            risk_val = float(self.var_risk_amount.get().strip())
            rr_val = float(self.var_rr.get().strip())
        except ValueError as e:
            self.log_message(f"[ERROR] Invalid format: {e}\n", "ERROR")
            return

        live_trading = self.mode_var.get() == "live"
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.log_area.delete("1.0", tk.END)
        self.log_message(f"[INFO] Initializing system in {'Live Trading' if live_trading else 'Backtest'} mode...\n", "INFO")
        
        # Build config dictionary for the bot engine
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
        
        # Stdout Redirection Setup
        self.log_stream = StringIO()
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream

        # Run Engine
        self.bot_thread = threading.Thread(target=self.run_bot_logic, args=(live_trading, config), daemon=True)
        self.bot_thread.start()
        
        # Start Log Polling Loop
        self.after(100, self.update_logs)

    def run_bot_logic(self, live_trading, config):
        try:
            run_trading_logic(live_trading, self.stop_event, config)
        except Exception as e:
            self.log_message(f"\n[ERROR] Thread execution failed: {e}\n", "ERROR")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def update_logs(self):
        log_contents = self.log_stream.getvalue()
        if log_contents:
            # Parse lines and apply beautiful tag coloring
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
            
            # Reset log stream
            self.log_stream.seek(0)
            self.log_stream.truncate(0)

        if self.bot_thread and self.bot_thread.is_alive():
            self.after(100, self.update_logs)
        else:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.log_message("\n[INFO] Engine process terminated.\n", "INFO")

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
