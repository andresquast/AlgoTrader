import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yaml
import logging
from enhanced_lstm import EnhancedLSTMPredictor  # Your LSTM model
from datetime import datetime, timedelta
import talib

class TradingSystem:
    def __init__(self, api_key, api_secret, config_path='config/config.yaml', paper=True):
        """Initialize the trading system with LSTM integration"""
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
        # Initialize LSTM model
        self.model = EnhancedLSTMPredictor(config_path=config_path)
        
        # Trading parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.stop_loss_atr_multiple = 2
        self.sequence_length = 60  # Match with LSTM sequence length
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_historical_data(self, symbol, lookback_days=100):
        """Get historical data with technical indicators"""
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        try:
            bars = self.data_client.get_stock_bars(request)
            df = pd.DataFrame(bars.data[symbol])
            
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            
            return df
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators for the model"""
        # Price columns
        df['PC_close'] = df['close']
        df['PC_high'] = df['high']
        df['PC_low'] = df['low']
        
        # Convert to numpy arrays for talib
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Technical indicators
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # MACD
        macd, signal, _ = talib.MACD(close)
        df['MACD'] = macd
        df['MACD_signal'] = signal
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close)
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower
        
        # Volume indicators
        df['OBV'] = talib.OBV(close, volume)
        
        # Momentum
        df['MOM'] = talib.MOM(close, timeperiod=10)
        
        # Drop any rows with NaN values
        df.dropna(inplace=True)
        
        return df

    def calculate_position_size(self, capital, current_price, atr):
        """Calculate position size based on risk management rules"""
        risk_amount = capital * self.risk_per_trade
        stop_loss = atr * self.stop_loss_atr_multiple
        shares = int(risk_amount / stop_loss)
        return min(shares, int(capital * 0.1 / current_price))  # Cap at 10% of capital

    def place_order(self, symbol, qty, side, stop_price=None):
        """Place a market order with optional stop loss"""
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        try:
            order = self.trading_client.submit_order(order_data)
            self.logger.info(f"Order placed: {qty} shares of {symbol} {side}")
            
            # Place stop loss if specified
            if stop_price and side == OrderSide.BUY:
                stop_order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price
                )
                self.trading_client.submit_order(stop_order)
                self.logger.info(f"Stop loss placed at {stop_price}")
                
            return order
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def run_strategy(self, symbol, confidence_threshold=0.7):
        """Execute trading strategy using LSTM predictions"""
        try:
            # Get historical data
            df = self.get_historical_data(symbol)
            if df is None or len(df) < self.sequence_length:
                return
                
            # Prepare data for LSTM
            X_train, X_val, y_train, y_val = self.model.prepare_data(df)
            
            # Train model if not trained
            if not hasattr(self.model, 'model') or self.model.model is None:
                self.logger.info("Training LSTM model...")
                self.model.train(X_train, y_train, X_val, y_val)
            
            # Get prediction
            latest_data = X_val[-1:] if len(X_val) > 0 else X_train[-1:]
            prediction = self.model.predict(latest_data)[0][0]
            
            # Get current position and account info
            try:
                position = self.trading_client.get_position(symbol)
                current_position = float(position.qty)
            except:
                current_position = 0
                
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)
            
            # Current market data
            current_price = df['close'].iloc[-1]
            current_atr = df['ATR'].iloc[-1]
            
            # Trading logic
            if prediction > confidence_threshold and current_position <= 0:
                # Calculate position size
                qty = self.calculate_position_size(buying_power, current_price, current_atr)
                if qty > 0:
                    stop_price = current_price - (current_atr * self.stop_loss_atr_multiple)
                    self.place_order(symbol, qty, OrderSide.BUY, stop_price)
                    
            elif prediction < -confidence_threshold and current_position >= 0:
                qty = abs(current_position) if current_position > 0 else \
                      self.calculate_position_size(buying_power, current_price, current_atr)
                if qty > 0:
                    self.place_order(symbol, qty, OrderSide.SELL)
            
            self.logger.info(f"""
            Prediction for {symbol}:
            Value: {prediction:.4f}
            Confidence Threshold: {confidence_threshold}
            Current Position: {current_position}
            ATR: {current_atr:.2f}
            """)
            
        except Exception as e:
            self.logger.error(f"Error in strategy execution: {e}")

if __name__ == "__main__":
    # Configuration
    API_KEY = "your_api_key_here"
    API_SECRET = "your_api_secret_here"
    CONFIG_PATH = "config/config.yaml"
    
    # Initialize trading system
    trader = TradingSystem(API_KEY, API_SECRET, CONFIG_PATH, paper=True)
    
    # Run strategy for specific symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    for symbol in symbols:
        trader.run_strategy(symbol)