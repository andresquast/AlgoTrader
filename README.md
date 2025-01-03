# AlgoTrader

A machine learning-based algorithmic trading system that integrates LSTM predictions with automated trading strategies.

## Project Structure

```
AlgoTrader/
├── src/
│   ├── data/          # Data collection and processing
│   ├── models/        # LSTM and trading models
│   └── utils/         # Helper functions
├── config/            # Configuration files
└── notebooks/         # Jupyter notebooks for analysis
```

## Environment Setup

1. Create and activate conda environment:

   ```bash
   conda create -n algotrading python=3.9
   conda activate algotrading
   ```

2. Install dependencies:

   ```bash
   # Install TensorFlow for Mac
   conda install -c apple tensorflow-deps
   python -m pip install tensorflow-macos tensorflow-metal

   # Install other dependencies
   conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn jupyter ipython pyyaml
   pip install alpaca-py yfinance
   conda install -c conda-forge ta-lib
   ```

## API Keys and Configuration

Store your API keys in config/keys.yaml. Never commit this file to version control!

Example keys.yaml structure:

```yaml
api_keys:
  alpha_vantage: "5O7VTPF7G6OFH4L4"
  alpaca:
    api_key: "your_alpaca_key"
    api_secret: "your_alpaca_secret"
```

To use the keys in your code:

```python
with open('config/keys.yaml', 'r') as file:
    keys = yaml.safe_load(file)
alpha_vantage_key = keys['api_keys']['alpha_vantage']
```

## Security Note

- Never commit API keys directly to version control
- Add config/keys.yaml to your .gitignore file
- Keep a template version of the keys file without actual keys

## Getting Started

1. Clone the repository
2. Set up the conda environment
3. Create config/keys.yaml with your API keys
4. Run the data collection scripts
5. Train the LSTM model
6. Start paper trading

## Important Commands

Start paper trading:

```bash
python src/models/trading_system.py
```

Collect new data:

```bash
python src/data/collector.py
```

## Model Details

The system uses:

- LSTM for price prediction
- Technical indicators for feature engineering
- Sentiment analysis from news data
- Risk management based on ATR

## Configuration

All model and trading parameters can be configured in config/config.yaml.

Key parameters include:

- Risk per trade
- Stop loss multipliers
- LSTM architecture
- Technical indicators
- Trading thresholds

## Data Sources

- Alpha Vantage: Market data and news sentiment
- [Add your other data sources]

## Risk Warning

This is an experimental trading system. Always:

- Start with paper trading
- Monitor system performance
- Use proper risk management
- Never risk money you can't afford to lose

## Next Steps

- [ ] Add more data sources
- [ ] Implement additional technical indicators
- [ ] Enhance risk management
- [ ] Add performance monitoring
- [ ] Implement portfolio management
