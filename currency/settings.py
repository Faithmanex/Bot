import os
import json
from .modules.trading_pairs import symbols

def get_project_root():
    """Returns the absolute path to the project's root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(relative_path):
    """Returns the absolute path for a given relative path from the project root."""
    return os.path.join(get_project_root(), relative_path)

def load_settings(settings_file="settings.json"):
    """Loads settings from a JSON file, applying default values for missing symbols."""
    settings_path = get_path(f'currency/{settings_file}')
    try:
        with open(settings_path, "r") as file:
            settings = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        settings = {}

    # Define default settings for symbols
    default_symbol_settings = {
        "polyorder": 6,
        "window_length": 7,
        "order": 5
    }

    # Merge defaults into every symbol already in the file
    # (covers sweep-saved symbols not in trading_pairs.symbols)
    all_symbols = set(symbols) | set(settings.keys())
    for symbol in all_symbols:
        if symbol not in settings:
            settings[symbol] = default_symbol_settings.copy()
        else:
            for key, val in default_symbol_settings.items():
                if key not in settings[symbol]:
                    settings[symbol][key] = val

    return settings

# Constants for directory paths
HISTORY_DATA_DIR = get_path("history_data")
BACKTEST_SUMMARY_DIR = get_path("backtest_summary")
ORDER_DB = get_path("sent_limits.csv")
