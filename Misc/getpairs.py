import MetaTrader5 as mt5

# Initialize connection to MT5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Get all available symbols
all_symbols = mt5.symbols_get()
for symbol in all_symbols:
    print(symbol.name)

# List of minor forex pairs (excluding major and exotic pairs)
minor_pairs = [
    "EURNOK", "EURNZD", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY",
    "GBPAUD", "GBPCHF", "GBPCAD", "GBPJPY", "GBPNOK",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDSGD", "AUDZAR",
    "CADCHF", "CADJPY", "NZDCAD", "CADMXN", "CADSEK",
    "CHFJPY", "CHFPLN", "CHFZAR",
    "NZDCAD", "NZDJPY",
    "EURSEK", "EURTRY",
    "GBPNZD", "GBPSGD", "GBPZAR",
    "AUDPLN", "AUDTRY",
    "CADPLN",
    "NZDCHF", "NZDSEK", "NZDSGD", "NZDTRY",
    "EURDKK", "EURHUF", "EURPLN", "EURSGD", "EURZAR",
    "GBPDKK", "GBPSEK", "GBPHUF", "GBPPLN", "GBPSGD", "GBPTRY", "GBPZAR",
    "AUDDKK", "AUDHUF", "AUDMXN", "AUDPLN", "AUDSEK", "AUDTRY", "AUDZAR",
    "CADDKK", "CADHUF", "CADNOK", "CADPLN", "CADTRY",
    "CHFSGD", "CHFNOK", "CHFMXN", "CHFSEK",
    "NZDDKK", "NZDHUF", "NZDNOK", "NZDPLN", "NZDTRY", "NZDZAR",
    "EURCZK", "EURMXN", "EURRUB", "EURHKD", "EURTHB",
    "GBPCZK", "GBPTHB", "GBPHKD", "GBPMXN", "GBPRUB",
    "AUDHKD", "AUDCZK", "AUDRUB", "AUDTHB",
    "CADHKD", "CADCZK", "CADTHB", "CADRUB", "CADZAR",
    "CHFTHB", "CHFRUB",
    "NZDMXN", "NZDTHB", "NZDRUB", "NZDHKD",
    "NOKJPY", "NOKSEK", "NOKCHF",
    "SEKJPY", "SEKCHF", "SEKZAR"
]

# Filter out the minor forex pairs available in MT5
available_minor_pairs = [symbol.name for symbol in all_symbols if symbol.name in minor_pairs]

# Print out the available minor forex pairs as a Python list
print("Available Minor Forex Pairs:")
print(available_minor_pairs)

# Shutdown connection to MT5
mt5.shutdown()
