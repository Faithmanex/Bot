import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
from io import StringIO
from currency.unified_trading import main as run_trading_logic

class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trading Bot")
        self.geometry("800x600")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        self.create_main_tab()
        self.create_logs_tab()

        self.stop_event = threading.Event()

    def create_main_tab(self):
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main")

        self.mode_var = tk.StringVar(value="backtest")

        ttk.Radiobutton(self.main_tab, text="Backtest", variable=self.mode_var, value="backtest").pack(pady=5)
        ttk.Radiobutton(self.main_tab, text="Live Trading", variable=self.mode_var, value="live").pack(pady=5)

        self.start_button = ttk.Button(self.main_tab, text="Start", command=self.start_bot)
        self.start_button.pack(pady=20)

        self.stop_button = ttk.Button(self.main_tab, text="Stop", command=self.stop_bot, state="disabled")
        self.stop_button.pack(pady=5)

    def create_logs_tab(self):
        self.logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_tab, text="Logs")
        self.log_area = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD, state="disabled")
        self.log_area.pack(pady=10, padx=10, expand=True, fill="both")

    def start_bot(self):
        live_trading = self.mode_var.get() == "live"
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.log_message("Bot started in {} mode.".format("Live Trading" if live_trading else "Backtest"))

        self.stop_event.clear()

        # Redirect stdout to a string stream
        self.log_stream = StringIO()
        sys.stdout = self.log_stream

        self.bot_thread = threading.Thread(target=self.run_bot_logic, args=(live_trading,), daemon=True)
        self.bot_thread.start()
        self.after(100, self.update_logs)

    def run_bot_logic(self, live_trading):
        try:
            run_trading_logic(live_trading, self.stop_event)
        except Exception as e:
            self.log_message(f"An error occurred: {e}")
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    def update_logs(self):
        log_contents = self.log_stream.getvalue()
        if log_contents:
            self.log_message(log_contents)
            # Clear the stream after reading
            self.log_stream.seek(0)
            self.log_stream.truncate(0)

        if self.bot_thread.is_alive():
            self.after(100, self.update_logs)
        else:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

    def stop_bot(self):
        self.stop_event.set()
        self.log_message("Stop signal sent. The bot will stop after the current operation.")
        self.stop_button.config(state="disabled")


    def log_message(self, message):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message)
        self.log_area.config(state="disabled")
        self.log_area.see(tk.END)

if __name__ == "__main__":
    app = TradingBotGUI()
    app.mainloop()
