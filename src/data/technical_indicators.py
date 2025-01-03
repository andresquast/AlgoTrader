import pandas as pd
import numpy as np
from typing import Optional, List, Dict

class TechnicalIndicators:
    def __init__(self):
        """Initialize Technical Indicators calculator"""
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate if dataframe has required columns"""
        return all(col in df.columns for col in self.required_columns)

    def calculate_sma(self, series: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate Simple Moving Average for multiple periods"""
        return {f'SMA_{period}': series.rolling(window=period).mean() 
                for period in periods}

    def calculate_ema(self, series: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate Exponential Moving Average for multiple periods"""
        return {f'EMA_{period}': series.ewm(span=period, adjust=False).mean() 
                for period in periods}

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Hist': macd_histogram
        }

    def calculate_bollinger_bands(self, prices: pd.Series, 
                                period: int = 20, 
                                std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        return {
            'BB_Upper': sma + (rolling_std * std_dev),
            'BB_Middle': sma,
            'BB_Lower': sma - (rolling_std * std_dev)
        }

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = pd.Series(index=df.index, dtype='float64')
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def calculate_stochastic(self, df: pd.DataFrame, 
                           k_period: int = 14, 
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return {
            'Stoch_K': k,
            'Stoch_D': d
        }

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various momentum indicators"""
        close = df['Close']
        
        momentum = {}
        # ROC - Rate of Change
        for period in [5, 10, 20]:
            momentum[f'ROC_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
            
        # Price Momentum
        for period in [5, 10, 20]:
            momentum[f'Momentum_{period}'] = close - close.shift(period)
        
        return momentum

    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        volume = df['Volume']
        close = df['Close']
        
        indicators = {}
        # Volume SMA
        for period in [5, 10, 20]:
            indicators[f'Volume_SMA_{period}'] = volume.rolling(window=period).mean()
        
        # Price-Volume Trend
        indicators['PVT'] = (((close - close.shift(1)) / close.shift(1)) * volume).cumsum()
        
        # Volume Rate of Change
        indicators['Volume_ROC'] = ((volume - volume.shift(10)) / volume.shift(10)) * 100
        
        return indicators

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        if not self._validate_dataframe(df):
            raise ValueError("DataFrame must contain OHLCV columns")
        
        df = df.copy()
        
        # Moving Averages
        sma_periods = [5, 10, 20, 50, 200]
        ema_periods = [5, 10, 20, 50, 200]
        
        for name, value in self.calculate_sma(df['Close'], sma_periods).items():
            df[name] = value
        
        for name, value in self.calculate_ema(df['Close'], ema_periods).items():
            df[name] = value
            
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        macd_data = self.calculate_macd(df['Close'])
        for name, value in macd_data.items():
            df[name] = value
            
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['Close'])
        for name, value in bb_data.items():
            df[name] = value
            
        # ATR
        df['ATR'] = self.calculate_atr(df)
        
        # OBV
        df['OBV'] = self.calculate_obv(df)
        
        # Stochastic Oscillator
        stoch_data = self.calculate_stochastic(df)
        for name, value in stoch_data.items():
            df[name] = value
            
        # Momentum Indicators
        momentum_data = self.calculate_momentum_indicators(df)
        for name, value in momentum_data.items():
            df[name] = value
            
        # Volume Indicators
        volume_data = self.calculate_volume_indicators(df)
        for name, value in volume_data.items():
            df[name] = value
            
        # Calculate volatility
        for period in [5, 10, 20]:
            df[f'Volatility_{period}'] = df['Close'].pct_change().rolling(window=period).std()
            
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df