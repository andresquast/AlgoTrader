from data.data_collection import DataCollector
import logging

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting data collection...")
        collector = DataCollector()
        data = collector.collect_all_data()
        logger.info("Data collection completed successfully")
        
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