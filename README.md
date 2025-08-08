# Options Predictor (Local, PyCharm)

## Quick Start
1. Open PyCharm → File > New Project from Existing Sources... and select this folder.
2. Create a virtual environment with Python 3.11+.
3. Open Terminal and run:
   ```bash
   pip install -r requirements.txt
   python -m src.train
   streamlit run app.py
   ```
4. Browser opens at http://localhost:8501

## How it works
- Fetches OHLCV from Yahoo Finance.
- Builds features like RSI, ATR, EMA distances.
- Trains LightGBM to predict 3-day up/down.
- Calibrates probabilities.
