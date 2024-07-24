import MetaTrader5 as mt5

# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Fetch all symbols from the broker
broker_symbols = mt5.symbols_get()

# Your predefined lists
major_pairs = [
    "XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "USDCHF"
]

minor_pairs = ['AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDJPY',
               'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
               'EURJPY', 'CADCHF', 'CADJPY', 'CHFJPY', 
               'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK', 
               'GBPCAD', 'GBPAUD', 'GBPCHF', 'GBPJPY', 
               'GBPNOK', 'GBPNZD', 'GBPSEK', 'NZDCAD', 
               'NZDJPY', 'AUDSGD', 'EURHKD', 'EURMXN', 
               'EURSGD', 'EURZAR', 'GBPSGD', 'NZDCHF', 
               'NZDSGD']


synthetic_pairs = [
    "Volatility 10 Index", "Volatility 10 (1s) Index", "Volatility 25 Index", 
    "Volatility 25 (1s) Index", "Volatility 50 Index", "Volatility 50 (1s) Index", 
    "Volatility 75 Index", "Volatility 75 (1s) Index", "Volatility 100 Index", 
    "Volatility 100 (1s) Index", "Volatility 150 (1s) Index", "Volatility 200 (1s) Index", 
    "Volatility 250 (1s) Index", "Volatility 300 (1s) Index", "Crash 300 Index", 
    "Crash 500 Index", "Crash 1000 Index", "Boom 300 Index", "Boom 500 Index", 
    "Boom 1000 Index", "Jump 10 Index", "Jump 25 Index", "Jump 50 Index", 
    "Jump 75 Index", "Jump 100 Index", "Drift Switch Index 10", "Drift Switch Index 20", 
    "Drift Switch Index 30", "DEX 600 UP Index", "DEX 900 UP Index", "DEX 1500 UP Index", 
    "DEX 600 DOWN Index", "DEX 900 DOWN Index", "DEX 1500 DOWN Index", 
    "Step Index", "Step 200 Index", "Step 500 Index"
]

# Function to check if a symbol matches a prefix and optionally has a suffix
def matches_with_prefix_and_optional_suffix(broker_symbol, expected_prefix):
    # Extract the base name of the broker symbol
    base_name = broker_symbol.name.split('.')[0]
    
    # Check if the base name starts with the expected prefix
    if base_name.startswith(expected_prefix):
        # Optionally, check for a suffix if needed
        # This is a simplistic check; adjust according to your needs
        expected_suffix = expected_prefix.split('_')[-1] if '_' in expected_prefix else ''
        actual_suffix = base_name.split('_')[-1] if '_' in base_name else ''
        
        return actual_suffix == expected_suffix
    return False

# Filter broker symbols based on your criteria
filtered_symbols = []
for symbol in broker_symbols:
    for pair_list in [major_pairs, minor_pairs, synthetic_pairs]:
        for pair in pair_list:
            if matches_with_prefix_and_optional_suffix(symbol, pair):
                filtered_symbols.append(symbol)

print(f"Filtered symbols: {len(filtered_symbols)}")
for symbol in filtered_symbols:
    print(symbol.name)

# Shut down MT5 connection
mt5.shutdown()
