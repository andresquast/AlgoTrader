import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import RobustScaler
from technical_indicators import TechnicalIndicators

class DataProcessor:
    def __init__(self, config_path: str = 'config/config.yaml', 
                 data_path: str = 'data/processed', 
                 data_saving_path: str = 'data/preprocessed'):
        """
        Initialize the DataProcessor.
        
        Args:
            config_path (str): Path to configuration file
            data_path (str): Path to input data directory
            data_saving_path (str): Path to save processed data
        """
        self.data_path = data_path
        self.data_saving_path = data_saving_path
        self.stock_scaler = RobustScaler()
        self.indicators = TechnicalIndicators()
        self.min_non_null_ratio = 0.95
        self.required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Create directories if they don't exist
        Path(data_saving_path).mkdir(parents=True, exist_ok=True)

    def validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logging.info(f"Validating data for {symbol}")
        
        # Handle date column
        date_column = None
        if 'Date' in df.columns:
            date_column = 'Date'
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        # Convert all numeric columns to float
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for null values
        null_ratio = df.isnull().sum() / len(df)
        valid_columns = null_ratio[null_ratio < (1 - self.min_non_null_ratio)].index
        df = df[valid_columns]
        
        # Remove rows with any remaining nulls
        df = df.dropna()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logging.info(f"Cleaned data shape for {symbol}: {df.shape}")
        return df

    def process_stock_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process stock data with technical indicators.
        
        Args:
            df (pd.DataFrame): Input stock data
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Processed stock data
        """
        try:
            # Add technical indicators
            df = self.indicators.add_all_indicators(df)
            
            # Forward fill any NaN values
            df = df.ffill()
            # Backward fill any remaining NaN values at the beginning
            df = df.bfill()
            
            # Final validation and cleaning
            df = self.validate_and_clean_data(df, symbol)
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing technical indicators for {symbol}: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and target variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled features and target variables
        """
        # Remove date column if it exists
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif df.index.name == 'Date':
            pass  # Index is already set to Date
        
        # Separate price columns and features
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in price_cols and col != 'Date']
        
        # Ensure all columns are numeric
        for col in feature_cols + price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.stock_scaler.fit_transform(df[feature_cols]),
            columns=feature_cols,
            index=df.index
        )
        
        # Create target variable (next day's return)
        target = df['Close'].pct_change().shift(-1)
        
        return scaled_features, target.dropna()

    def process_data(self, symbols: list) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process data for all symbols.
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Processed data for each symbol
        """
        processed_data = {}
        
        for symbol in symbols:
            try:
                # Load data
                stock_path = f"{self.data_path}/{symbol}_stock_processed.csv"
                df = pd.read_csv(stock_path)
                
                # Process stock data
                processed_df = self.process_stock_data(df, symbol)
                
                # Handle date column
                if 'Date' in processed_df.columns:
                    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
                    processed_df.set_index('Date', inplace=True)
                
                # Prepare features and target
                features, target = self.prepare_features(processed_df)
                
                # Remove last row since it won't have a target value
                features = features[:-1]
                target = target[:-1]
                
                processed_data[symbol] = {
                    'features': features,
                    'target': target,
                    'raw_data': processed_df[:-1]
                }
                
                # Save processed data
                save_path = f"{self.data_saving_path}/{symbol}_processed.csv"
                processed_df.to_csv(save_path)
                logging.info(f"Saved processed data for {symbol} to {save_path}")
                
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        return processed_data

    def create_sequences(self, features: pd.DataFrame, target: pd.Series, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model.
        
        Args:
            features (pd.DataFrame): Feature data
            target (pd.Series): Target data
            sequence_length (int): Length of sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X, y = [], []
        features_array = features.values
        target_array = target.values
        
        # Adjust the range to prevent index out of bounds
        for i in range(len(features_array) - sequence_length):
            sequence = features_array[i:(i + sequence_length)]
            target_value = target_array[i + sequence_length - 1]  # Changed from i + sequence_length
            
            # Only add the sequence if it's complete
            if len(sequence) == sequence_length:
                X.append(sequence)
                y.append(target_value)
        
        return np.array(X), np.array(y)

    def prepare_lstm_data(self, processed_data: Dict[str, Dict[str, pd.DataFrame]], 
                         sequence_length: int = 60, 
                         train_split: float = 0.8) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare data for LSTM model.
        
        Args:
            processed_data (Dict): Processed data for each symbol
            sequence_length (int): Length of sequences
            train_split (float): Train/validation split ratio
            
        Returns:
            Dict[str, Dict[str, np.ndarray]]: LSTM-ready data for each symbol
        """
        lstm_data = {}
        
        for symbol, data in processed_data.items():
            try:
                features = data['features']
                target = data['target']
                
                logging.info(f"Creating sequences for {symbol} with shape: {features.shape}")
                
                # Create sequences
                X, y = self.create_sequences(features, target, sequence_length)
                
                if len(X) == 0:
                    logging.warning(f"No sequences created for {symbol}")
                    continue
                
                # Split into train and validation sets
                train_size = int(len(X) * train_split)
                
                lstm_data[symbol] = {
                    'X_train': X[:train_size],
                    'X_val': X[train_size:],
                    'y_train': y[:train_size],
                    'y_val': y[train_size:]
                }
                
                logging.info(f"Successfully created LSTM data for {symbol}")
                logging.info(f"X_train shape: {X[:train_size].shape}")
                logging.info(f"X_val shape: {X[train_size:].shape}")
                
            except Exception as e:
                logging.error(f"Error preparing LSTM data for {symbol}: {str(e)}")
                continue
        
        return lstm_data

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    symbols = ['AAPL', 'GOOG', 'MSFT']
    processed_data = processor.process_data(symbols)
    lstm_data = processor.prepare_lstm_data(processed_data)