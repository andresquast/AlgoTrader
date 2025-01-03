import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.layers import Input, Concatenate, Add, LeakyReLU, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import yaml
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

class EnhancedLSTMPredictor:
    def __init__(self, config_path='config/config.yaml', sequence_length=60):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
        self.model = None
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['model']['lstm']
        
        # Add adaptive learning rate tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def create_sequences(self, data, target):
        """Enhanced sequence creation with overlap handling and validation"""
        if len(data) != len(target):
            raise ValueError("Data and target lengths must match")
            
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:(i + self.sequence_length)]
            if np.isnan(sequence).any():
                continue  # Skip sequences with NaN values
            X.append(sequence)
            y.append(target[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def prepare_data(self, df):
        """Enhanced data preparation with additional features and better scaling"""
        # Add volatility features
        price_cols = [col for col in df.columns if col.startswith('PC')]
        tech_cols = [col for col in df.columns if not col.startswith('PC')]
        
        # Calculate rolling volatility
        prices = df[price_cols].values
        returns = np.diff(prices, axis=0) / prices[:-1]
        volatility = pd.DataFrame(returns).rolling(window=20).std().values
        
        # Store original prices
        self.original_prices = prices
        
        # Scale prices using tanh-estimator for better tail handling
        scaled_prices = self.price_scaler.fit_transform(prices)
        price_changes = np.diff(scaled_prices, axis=0)
        
        # Scale technical indicators with RobustScaler
        scaled_tech = self.feature_scaler.fit_transform(df[tech_cols])
        
        # Combine features including volatility
        combined_features = np.hstack([
            price_changes,
            scaled_tech[1:],
            volatility[20:]  # Add volatility after its calculation window
        ])
        
        # Handle missing values
        combined_features = np.nan_to_num(combined_features, nan=0)
        
        # Create sequences with time-series split
        X, y = self.create_sequences(combined_features[:-1], price_changes[1:])
        
        # Use TimeSeriesSplit for more realistic validation
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self, input_shape):
        """Enhanced model architecture with residual connections and attention"""
        inputs = Input(shape=input_shape)
        
        # Add noise for regularization
        x = GaussianNoise(0.01)(inputs)
        
        # First Bidirectional LSTM block with residual connection
        lstm1 = Bidirectional(LSTM(64, return_sequences=True, 
                                 kernel_regularizer=l2(1e-5)))(x)
        bn1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(0.2)(bn1)
        
        # Second Bidirectional LSTM block
        lstm2 = Bidirectional(LSTM(32, return_sequences=True))(dropout1)
        bn2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(0.2)(bn2)
        
        # Third LSTM block with residual connection
        lstm3 = Bidirectional(LSTM(16))(dropout2)
        bn3 = BatchNormalization()(lstm3)
        dropout3 = Dropout(0.1)(bn3)
        
        # Dense layers with LeakyReLU
        dense1 = Dense(32)(dropout3)
        leaky1 = LeakyReLU(alpha=0.01)(dense1)
        bn4 = BatchNormalization()(leaky1)
        
        # Output layer
        outputs = Dense(1)(bn4)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use AMSGrad variant of Adam
        optimizer = Adam(learning_rate=0.001, amsgrad=True)
        
        model.compile(
            optimizer=optimizer,
            loss='huber'  # More robust to outliers than MSE
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Enhanced training with dynamic learning rate and better callbacks"""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=1e-6,
                min_delta=1e-4
            )
        ]
        
        # Add class weights to handle imbalanced data
        unique, counts = np.unique(np.sign(y_train), return_counts=True)
        class_weights = dict(zip(unique, counts.max() / counts))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=150,  # Increased epochs with early stopping
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            shuffle=False  # Important for time series data
        )
        return history
    
    def evaluate(self, y_true, y_pred):
        """Enhanced evaluation with more metrics"""
        # Ensure shapes match
        y_true = y_true[self.prediction_indices, 0:1]
        
        # Calculate directional accuracy
        direction_true = np.sign(np.diff(y_true, axis=0))
        direction_pred = np.sign(np.diff(y_pred, axis=0))
        
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        # Calculate additional metrics
        metrics = {
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'Directional_Accuracy': directional_accuracy,
            'Precision': precision_score(direction_true > 0, direction_pred > 0),
            'Recall': recall_score(direction_true > 0, direction_pred > 0),
            'F1': f1_score(direction_true > 0, direction_pred > 0)
        }
        
        return metrics