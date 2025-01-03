# src/data/data_cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from textblob import TextBlob
from datetime import datetime
import nltk
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path=None):
    """
    Load configuration from yaml file.

    Args:
        config_path (str, optional): Path to config file. If None, uses default path.
    """
    if config_path is None:
        # Use default path relative to project root
        file_dir = Path(__file__).parent.parent.parent
        config_path = file_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def clean_stock_data(df, config):
    """
    Clean and preprocess stock price data.
    """
    logger.info("Cleaning stock data...")
    
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # Convert Date to datetime and normalize to midnight UTC
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Handle missing values
    df = df.dropna()
    
    # Calculate technical indicators based on config
    tech_indicators = config['features']['technical_indicators']
    
    # Calculate returns
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    for indicator in tech_indicators:
        if indicator == 'SMA':
            # Simple Moving Averages
            df['SMA5'] = df.groupby('Ticker')['Close'].rolling(window=5).mean().reset_index(0, drop=True)
            df['SMA20'] = df.groupby('Ticker')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
        
        elif indicator == 'RSI':
            # Relative Strength Index
            df['RSI'] = df.groupby('Ticker')['Close'].apply(calculate_rsi).reset_index(0, drop=True)
        
        elif indicator == 'MACD':
            # MACD and Signal line
            for ticker in df['Ticker'].unique():
                mask = df['Ticker'] == ticker
                macd, signal = calculate_macd(df.loc[mask, 'Close'])
                df.loc[mask, 'MACD'] = macd
                df.loc[mask, 'MACD_Signal'] = signal
        
        elif indicator == 'Volatility':
            # 20-day rolling volatility
            df['Volatility'] = df.groupby('Ticker')['Returns'].rolling(window=20).std().reset_index(0, drop=True)
        
        elif indicator == 'Volume_Change':
            # Volume changes
            df['Volume_Change'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # Create target variable (1 if price goes up tomorrow, 0 if down)
    df['Target'] = df.groupby('Ticker')['Close'].shift(-1) > df['Close']
    df['Target'] = df['Target'].astype(int)
    
    # Drop rows with NaN values created by rolling calculations
    df = df.dropna()
    
    return df

def clean_news_data(df, config):
    """
    Clean and preprocess news data.
    """
    logger.info("Cleaning news data...")
    
    if not config['features']['sentiment_analysis']['enabled']:
        logger.info("Sentiment analysis disabled in config")
        return None
    
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # Convert time_published to datetime with UTC timezone
    df['time_published'] = pd.to_datetime(df['time_published'], utc=True)
    
    # Extract date and keep UTC timezone
    df['Date'] = df['time_published'].dt.normalize()
    
    # Calculate sentiment scores using TextBlob if sentiment not provided
    if 'sentiment' not in df.columns:
        logger.info("Calculating sentiment scores...")
        df['sentiment_score'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Group by date and calculate daily sentiment metrics
    daily_sentiment = df.groupby('Date').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = ['Date', 'avg_sentiment', 'sentiment_std', 'news_count']
    
    return daily_sentiment

def combine_data(stock_df, news_df):
    """
    Combine stock and news data.
    """
    logger.info("Combining stock and news data...")
    
    if news_df is not None:
        # Create copies to avoid modifying original data
        stock_df = stock_df.copy()
        news_df = news_df.copy()
        
        # Convert both dates to UTC and then remove timezone info
        stock_df['Date'] = stock_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        news_df['Date'] = news_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Merge stock and news data
        combined_df = pd.merge(stock_df, news_df, on='Date', how='left')
        
        # Fill missing news data with 0 (days without news)
        news_columns = ['avg_sentiment', 'sentiment_std', 'news_count']
        combined_df[news_columns] = combined_df[news_columns].fillna(0)
    else:
        combined_df = stock_df
        # Remove timezone info from stock data if no news data
        combined_df['Date'] = combined_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    return combined_df

def main():
    """Main function to run data cleaning pipeline."""
    config = load_config()
    
    try:
        # Initialize paths
        raw_data_path = Path("data/raw")
        processed_data_path = Path("data/processed")
        
        # Get list of stock files
        stock_files = list(raw_data_path.glob("*_stock_data.csv"))
        
        if not stock_files:
            logger.error("No stock data files found in data/raw directory")
            return
        
        # Process each stock file
        all_stock_data = []
        all_news_data = []
        
        for stock_file in stock_files:
            # Extract ticker from filename
            ticker = stock_file.name.split('_')[0]
            logger.info(f"Processing data for {ticker}")
            
            # Load and process stock data
            stock_df = pd.read_csv(stock_file)
            stock_df['Ticker'] = ticker  # Add ticker column if not present
            all_stock_data.append(stock_df)
            
            # Check for corresponding news file
            news_file = raw_data_path / f"{ticker}_news_data.csv"
            if news_file.exists() and config['features']['sentiment_analysis']['enabled']:
                news_df = pd.read_csv(news_file)
                all_news_data.append(news_df)
        
        # Combine all stock data
        if all_stock_data:
            combined_stock_df = pd.concat(all_stock_data, ignore_index=True)
            cleaned_stock_df = clean_stock_data(combined_stock_df, config)
            logger.info("Stock data cleaned successfully")
        else:
            logger.error("No stock data to process")
            return
        
        # Process news data if available
        if all_news_data and config['features']['sentiment_analysis']['enabled']:
            combined_news_df = pd.concat(all_news_data, ignore_index=True)
            cleaned_news_df = clean_news_data(combined_news_df, config)
            logger.info("News data cleaned successfully")
        else:
            cleaned_news_df = None
            logger.info("No news data to process")
        
        # Combine datasets
        final_df = combine_data(cleaned_stock_df, cleaned_news_df)
        
        # Save processed data
        output_path = processed_data_path / "combined_processed_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        # Print data info
        logger.info("\nProcessed Data Info:")
        logger.info(f"Shape: {final_df.shape}")
        logger.info("\nFeatures created:")
        for indicator in config['features']['technical_indicators']:
            logger.info(f"- {indicator}")
        if config['features']['sentiment_analysis']['enabled']:
            logger.info("\nSentiment features:")
            for metric in config['features']['sentiment_analysis']['metrics']:
                logger.info(f"- {metric}")
        
    except Exception as e:
        logger.error(f"Error in data cleaning pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()