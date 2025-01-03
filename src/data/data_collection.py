"""
Data collection module using Alpaca API for stock market prediction project.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = self._load_config(config_path)
        self.keys = self._load_keys('config/keys.yaml')
        
        try:
            # Access data configuration
            self.symbols = self.config.get('data', {}).get('ticker_list', [])
            self.start_date = self.config.get('data', {}).get('start_date')
            self.end_date = self.config.get('data', {}).get('end_date')
            
            # Initialize Alpaca clients
            self.historical_client = StockHistoricalDataClient(
                api_key=self.keys['alpaca']['api_key'],
                secret_key=self.keys['alpaca']['secret_key']
            )
            
            # Initialize paths
            self.raw_data_path = Path('data/raw')
            self.processed_data_path = Path('data/processed')
            self._setup_directories()
            
        except Exception as e:
            logger.error(f"Error initializing DataCollector: {str(e)}")
            raise

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def _load_keys(keys_path: str) -> dict:
        with open(keys_path, 'r') as file:
            return yaml.safe_load(file)

    def _setup_directories(self):
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical stock data for a single symbol with split adjustments.
        """
        logger.info(f"Fetching stock data for {symbol}")
        try:
            # Create request parameters with adjustment flag
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=datetime.strptime(self.start_date, '%Y-%m-%d'),
                end=datetime.strptime(self.end_date, '%Y-%m-%d'),
                adjustment='split'  # This ensures prices are adjusted for splits
            )
            
            # Get the data
            bars = self.historical_client.get_stock_bars(request_params)
            logger.info(f"Raw data type: {type(bars)}")
            
            # Convert to DataFrame
            df = bars.df
            
            # If multi-index DataFrame, get the specific symbol data
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol)
            
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame shape: {df.shape}")
            
            # Rename columns to match our convention
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'trade_count': 'TradeCount',
                'vwap': 'VWAP'
            })
            
            # Add basic price features
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Save raw data with note about split adjustment
            output_path = self.raw_data_path / f"{symbol}_stock_data_adjusted.csv"
            df.to_csv(output_path)
            logger.info(f"Saved split-adjusted data to {output_path}. Shape: {df.shape}")
            
            # Log some basic statistics to verify the adjustment
            logger.info(f"Price range for {symbol}:")
            logger.info(f"Min Close: ${df['Close'].min():.2f}")
            logger.info(f"Max Close: ${df['Close'].max():.2f}")
            logger.info(f"Current Close: ${df['Close'].iloc[-1]:.2f}")
            
            return df
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect stock data for all symbols."""
        all_data = {}
        
        for symbol in self.symbols:
            logger.info(f"Processing data for {symbol}")
            stock_data = self.fetch_stock_data(symbol)
            
            if not stock_data.empty:
                all_data[symbol] = stock_data
                processed_path = self.processed_data_path / f"{symbol}_stock_processed.csv"
                stock_data.to_csv(processed_path)
                logger.info(f"Saved processed data to {processed_path}")
        
        return all_data

def test_alpaca_connection():
    """Test the Alpaca API connection."""
    try:
        with open('config/keys.yaml', 'r') as file:
            keys = yaml.safe_load(file)
        
        client = StockHistoricalDataClient(
            api_key=keys['alpaca']['api_key'],
            secret_key=keys['alpaca']['secret_key']
        )
        
        # Try to fetch a single day of AAPL data as a test
        request_params = StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=TimeFrame.Day,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2)
        )
        
        test_data = client.get_stock_bars(request_params)
        logger.info("Successfully connected to Alpaca API")
        logger.info(f"Test data type: {type(test_data)}")
        logger.info(f"Test data structure: {test_data}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {str(e)}")
        return False

def main():
    """Main function to run data collection."""
    try:
        # Test connection first
        if not test_alpaca_connection():
            logger.error("Failed Alpaca connection test")
            return
            
        collector = DataCollector()
        data = collector.collect_all_data()
        
        # Print summary of collected data
        for symbol, df in data.items():
            logger.info(f"\nData collected for {symbol}:")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()