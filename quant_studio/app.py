import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor

app = Flask(__name__, template_folder='templates')
CORS(app)

DATA_DIR = os.path.join(_ROOT, 'data')

# Load Kronos model once at startup
print("[Kronos] Loading model...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(model, tokenizer, max_context=2048)
print("[Kronos] Model ready on port 7080.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        filename = data.get('filename', 'NSE_NIFTY50_5m.csv')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))

        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {filename}'}), 400

        df = pd.read_csv(file_path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
        df['amount'] = pd.to_numeric(df.get('amount', df['close'] * df['volume']), errors='coerce').fillna(0)
        df = df.dropna().reset_index(drop=True)

        if len(df) < lookback + pred_len:
            return jsonify({'error': f'Dataset too short ({len(df)} rows)'}), 400

        # Latest window with ground truth for comparison
        s_idx = len(df) - (lookback + pred_len)
        slice_df = df.iloc[s_idx : s_idx + lookback + pred_len].reset_index(drop=True)

        x_df = slice_df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_ts = slice_df.loc[:lookback-1, 'timestamps']
        y_ts = slice_df.loc[lookback:lookback+pred_len-1, 'timestamps']
        actual_df = slice_df.loc[lookback:lookback+pred_len-1].reset_index(drop=True)

        # Predict with Kronos
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )

        # Simple accuracy
        mae = float(np.mean(np.abs(actual_df['close'].values - pred_df['close'].values)))
        rmse = float(np.sqrt(np.mean((actual_df['close'].values - pred_df['close'].values) ** 2)))
        act_dir = np.sign(np.diff(np.insert(actual_df['close'].values, 0, x_df['close'].iloc[-1])))
        pred_dir = np.sign(np.diff(np.insert(pred_df['close'].values, 0, x_df['close'].iloc[-1])))
        hit_rate = float(np.mean(act_dir == pred_dir) * 100)

        return jsonify({
            'success': True,
            'stats': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'hit_rate': round(hit_rate, 1)
            },
            'history': {
                'close': [float(x) for x in x_df['close']],
                'volume': [float(x) for x in x_df['volume']]
            },
            'actual': {
                'close': [float(x) for x in actual_df['close']],
                'volume': [float(x) for x in actual_df['volume']]
            },
            'prediction': {
                'close': [float(x) for x in pred_df['close']],
                'volume': [float(x) for x in pred_df['volume']]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7080, use_reloader=False)
