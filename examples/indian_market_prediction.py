import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor

def plot_prediction(kline_df, pred_df, title="Kronos Prediction on Indian Market (NSE NIFTY 50)"):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price (INR)', fontsize=13)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=13)
    ax2.set_xlabel('Time Step / Index', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "indian_market_prediction.png")
    plt.savefig(output_path, dpi=150)
    print(f"Chart successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    print("1. Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)

    print("2. Preparing Indian Dataset (NSE NIFTY 50 5min)...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "NSE_NIFTY50_5m.csv")
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    lookback = 400
    pred_len = 120

    # Pick the most recent test slice
    s_idx = len(df) - (lookback + pred_len)
    slice_df = df.iloc[s_idx : s_idx + lookback + pred_len].reset_index(drop=True)

    x_df = slice_df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = slice_df.loc[:lookback-1, 'timestamps']
    y_timestamp = slice_df.loc[lookback:lookback+pred_len-1, 'timestamps']

    print(f"3. Running Kronos forecast ({pred_len} steps into future)...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    print("Forecasted Data Head:")
    print(pred_df.head())

    kline_df = slice_df.loc[:lookback+pred_len-1]
    plot_prediction(kline_df, pred_df, title="Kronos Prediction vs Ground Truth (NSE NIFTY 50 5m)")
