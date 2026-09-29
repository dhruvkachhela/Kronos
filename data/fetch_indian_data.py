import os
import sys
import pandas as pd
import numpy as np

def fetch_and_format_ticker(ticker_symbol, name, interval="5m", period="60d", output_dir="data"):
    """
    Fetch data for an Indian stock or index using yfinance and save in Kronos format.
    Columns: timestamps, open, high, low, close, volume, amount
    """
    import yfinance as yf
    
    print(f"Fetching {ticker_symbol} ({name}) interval={interval} period={period}...")
    try:
        t = yf.Ticker(ticker_symbol)
        df = t.history(interval=interval, period=period, auto_adjust=False)
        if df.empty:
            print(f"Warning: No data returned for {ticker_symbol} with interval={interval}, period={period}")
            # Try fallback period
            if interval == "5m":
                df = t.history(interval=interval, period="30d", auto_adjust=False)
        
        if df.empty:
            print(f"Error: Could not retrieve data for {ticker_symbol}")
            return None
        
        df = df.reset_index()
        # Find datetime column
        dt_col = None
        for col in ['Datetime', 'Date', 'index', 'timestamp', 'timestamps']:
            if col in df.columns:
                dt_col = col
                break
        
        if dt_col is None:
            dt_col = df.columns[0]
            
        df['timestamps'] = pd.to_datetime(df[dt_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Rename standard columns
        col_map = {}
        for col in df.columns:
            c_lower = str(col).lower()
            if 'open' in c_lower:
                col_map[col] = 'open'
            elif 'high' in c_lower:
                col_map[col] = 'high'
            elif 'low' in c_lower:
                col_map[col] = 'low'
            elif 'close' in c_lower and 'adj' not in c_lower:
                col_map[col] = 'close'
            elif 'volume' in c_lower:
                col_map[col] = 'volume'
        
        df = df.rename(columns=col_map)
        
        # Fill missing required columns
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                print(f"Error: Missing column {col}")
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        else:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            
        df['amount'] = df['close'] * df['volume']
        
        # Keep clean columns in Kronos order
        clean_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[clean_cols].dropna().reset_index(drop=True)
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{name}_{interval}.csv"
        out_path = os.path.join(output_dir, filename)
        df.to_csv(out_path, index=False)
        print(f"Successfully saved {len(df)} rows to {out_path}")
        return out_path
    except Exception as e:
        print(f"Exception fetching {ticker_symbol}: {e}")
        return None

def fetch_default_indian_suite(output_dir="data"):
    # Prominent Indian Assets
    suite = [
        ("^NSEI", "NSE_NIFTY50", "5m", "60d"),
        ("^NSEBANK", "NSE_BANKNIFTY", "5m", "60d"),
        ("RELIANCE.NS", "NSE_RELIANCE", "5m", "60d"),
        ("TCS.NS", "NSE_TCS", "5m", "60d"),
        ("HDFCBANK.NS", "NSE_HDFCBANK", "5m", "60d"),
        ("INFY.NS", "NSE_INFY", "5m", "60d"),
        ("TATAMOTORS.NS", "NSE_TATAMOTORS", "5m", "60d"),
        ("^NSEI", "NSE_NIFTY50", "1d", "2y"),
        ("RELIANCE.NS", "NSE_RELIANCE", "1d", "2y"),
    ]
    
    saved = []
    for sym, name, interval, period in suite:
        path = fetch_and_format_ticker(sym, name, interval=interval, period=period, output_dir=output_dir)
        if path:
            saved.append(path)
    return saved

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
    fetch_default_indian_suite(output_dir=out_dir)
