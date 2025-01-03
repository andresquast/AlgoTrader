from data_preprocessing import DataProcessor
import logging
import argparse
import os
import numpy as np

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create data directories if they don't exist
    os.makedirs('data/preprocessed', exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOG', 'MSFT'],
                       help='List of stock symbols to process')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Sequence length for LSTM data')
    args = parser.parse_args()

    # Initialize processor
    processor = DataProcessor()
    
    try:
        # Process data
        logging.info("Starting data processing...")
        processed_data = processor.process_data(args.symbols)
        
        if not processed_data:
            logging.error("No data was processed successfully")
            return
        
        # Prepare LSTM data
        logging.info("Preparing LSTM sequences...")
        lstm_data = processor.prepare_lstm_data(processed_data, 
                                              sequence_length=args.sequence_length)
        
        # Print results
        for symbol in args.symbols:
            if symbol in lstm_data:
                data = lstm_data[symbol]
                logging.info(f"\nProcessed data shapes for {symbol}:")
                logging.info(f"X_train shape: {data['X_train'].shape}")
                logging.info(f"y_train shape: {data['y_train'].shape}")
                logging.info(f"X_val shape: {data['X_val'].shape}")
                logging.info(f"y_val shape: {data['y_val'].shape}")
                
                # Print some basic statistics
                logging.info(f"\nTarget statistics for {symbol}:")
                logging.info(f"Training target mean: {np.mean(data['y_train']):.4f}")
                logging.info(f"Training target std: {np.std(data['y_train']):.4f}")
            else:
                logging.warning(f"No processed data available for {symbol}")
            
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()