import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureReducer:
    def __init__(self, stock_dict, n_components_stock, n_components_news):
        self.stock_data = stock_dict
        self.n_components_stock = n_components_stock
        self.n_components_news = n_components_news
        self.pca = PCA()    # Only Feature reducing algorithm used currently is PCA, but more could be added below 

    def use_pca(self):
        """
        Preprocess stock predictor data with PCA on both stock data and news data.
        
        Args:
            stock_dict (dict): Dictionary with stock symbols as keys and a nested dictionary containing
                                'stock_data' and 'news_data' (both are pd.DataFrame).
            n_components_stock: Number of principal components for stock data.
            n_components_news: Number of principal components for news data.
        
        Updates:
            self.stock_dict - dict: Dictionary with PCA-transformed 'stock_data' and 'news_data'.
        """
        for stock, data in self.stock_data.items():
            stock_data = data['stock_data']
            news_data = data['news_data']

            stock_data = stock_data.fillna(0)
            news_data = news_data.fillna(0)

            # Standardize data for PCA
            numerical_stock_data = self.convert_date(stock_data)
            numerical_news_data = self.make_news_numerical(news_data)
            stock_scaler = StandardScaler()
            news_scaler = StandardScaler()
            stock_data_scaled = stock_scaler.fit_transform(numerical_stock_data)
            news_data_scaled = news_scaler.fit_transform(numerical_news_data)

            # Apply PCA on data
            pca_stock = PCA(n_components = self.n_components_stock)
            pca_news = PCA(n_components = self.n_components_news)
            stock_data_pca = pca_stock.fit_transform(stock_data_scaled)
            news_data_pca = pca_news.fit_transform(news_data_scaled)
            stock_data_pca = pd.DataFrame(stock_data_pca, columns=[f'PC{i+1}_stock' for i in range(self.n_components_stock)])
            news_data_pca = pd.DataFrame(news_data_pca, columns=[f'PC{i+1}_news' for i in range(self.n_components_news)])

            # Replace current data with new PCA data
            self.stock_data[stock]['stock_data'] = stock_data_pca
            self.stock_data[stock]['news_data'] = news_data_pca

    """
    Numerical data is necessary for PCA to run, so the datetime portion of stock data is converted into numbers
    """
    def convert_date(self, stock_data):
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True)
        stock_data['DayOfWeek'] = pd.to_datetime(stock_data['Date'], utc=True).dt.dayofweek
        stock_data['DayOfMonth'] = pd.to_datetime(stock_data['Date'], utc=True).dt.day
        stock_data['Month'] = pd.to_datetime(stock_data['Date'], utc=True).dt.month
        stock_data['Year'] = pd.to_datetime(stock_data['Date'], utc=True).dt.year
        stock_data['Quarter'] = pd.to_datetime(stock_data['Date'], utc=True).dt.quarter
        stock_data['Hour'] = pd.to_datetime(stock_data['Date'], utc=True).dt.hour
        stock_data = stock_data.drop(columns=['Date'])
        return stock_data
    
    """
    Numerical data is necessary for PCA to run, so get all numerical datapoints out of news data
    """
    def make_news_numerical(self, news_data):
        numerical_news = news_data
        numerical_news = numerical_news.drop(columns= ['title', 'url', 'time_published', 'authors', 'summary', 'banner_image',
                                                       'source', 'category_within_source', 'source_domain', 'topics',
                                                       'overall_sentiment_label', 'ticker_sentiment', 'cleaned_text',
                                                       'sentiment'])
        return numerical_news

    """
    Getter for stock_data once its dimensions have been reduced

    Returns:
        self.stock_data - dict: Dictionary with dimension reduced stock data 
    """
    def get_stock_data(self):
        return self.stock_data
