# Set the conditions
fx = True
synthetics = False
all_pairs = True
major = False
minor = False
single_symbol = False  # Flag for selecting a single symbol

# Define the forex pairs
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

# Function to determine the single symbol based on conditions or user input
def determine_symbol():
    if fx:
        if all_pairs:
            return major_pairs + minor_pairs
        elif major:
            return major_pairs
        elif minor:
            return minor_pairs
        else:
            raise ValueError("No symbols selected. Please check the variables.")
    elif synthetics:
        return synthetic_pairs
    else:
        raise ValueError("No symbols selected. Please check the variables.")

# Determine the symbols
try:
    if single_symbol:
        # Example: Selecting EURUSD based on some condition or input
        # selected_symbols = ["EURNOK", "EURNZD", "GBPNOK", "EURHKD", "EURGBP"] # Replace with your logic to determine the symbol dynamically
        selected_symbols = [                            
                            'ADAUSD', 'DOGUSD', 'DSHUSD', 'ETCUSD', 'IOTUAD', 'XMRUSD', 'XPRUSD', 'BNBUSD'
                            ]

        # Replace with your logic to determine the symbol dynamically
        symbols = selected_symbols
    else:
        symbols = determine_symbol()
    print(f"Selected symbols: {symbols}")

except ValueError as e:
    print(e)
    symbols = []

import MetaTrader5 as mt5
mt5.initialize()
broker_symbols = mt5.symbols_get()

matched_symbols = []

for broker_symbol in broker_symbols:
    for my_symbol in symbols:
        if broker_symbol.name.startswith(my_symbol):
            matched_symbols.append(broker_symbol.name)
        else:
            # Check for symbols with suffixes
            if '.' in broker_symbol.name:
                base, suffix = broker_symbol.name.split('.', 1)
                if base == my_symbol:
                    matched_symbols.append(broker_symbol.name)

symbols = matched_symbols
