import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        #self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.closed = False

        
class TradingStrategy:
    def __init__(self, file):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.com = 0.00125
        self.strategy_value = [1_000_000]
        self.n_shares = 10
        self.file = file
        self.file_mapping = {
            "5m": "data/aapl_5m_train.csv",
            "1h": "data/aapl_1h_train.csv",
            "1d": "data/aapl_1d_train.csv",
            "1m": "data/aapl_1m_train.csv"
        }
        self.load_data(self.file)
        
        self.active_indicators = []
        
        self.calculate_new_features()
        
        
        self.best_combination = None
        self.best_value = 0
        self.stop_loss = 0.95
        self.take_profit = 1.05
        self.n_shares = 10        
        self.best_buylog_params = None
        self.best_selllog_params = None
    
    @staticmethod
    def get_slope(series):
        y = series.values.reshape(-1, 1)
        X = np.array(range(len(series))).reshape(-1, 1)
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        return lin_reg.coef_[0][0]
    
    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)
        
    def calculate_new_features(self):
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=10).std()
        self.data['Close_Trend'] = self.data['Close'].rolling(window=10).apply(self.get_slope, raw=False)
        self.data['Volume_Trend'] = self.data['Volume'].rolling(window=10).apply(self.get_slope, raw=False)
        self.data['Spread'] = self.data['High'] - self.data['Low']
        self.data['Future_Return_Avg_5'] = self.data['Returns'].shift(-1).rolling(window=10).mean().shift(-9)
        threshold_buy = self.data['Future_Return_Avg_5'].quantile(0.85) 
        threshold_sell = self.data['Future_Return_Avg_5'].quantile(0.15)
        self.data['LR_Buy_Signal'] = (self.data['Future_Return_Avg_5'] > threshold_buy).astype(int)
        self.data['LR_Sell_Signal'] = (self.data['Future_Return_Avg_5'] < threshold_sell).astype(int)
        self.data['Pt-1'] = self.data['Close'].shift(1)
        self.data['Pt-2'] = self.data['Close'].shift(2)
        self.data['Pt-3'] = self.data['Close'].shift(3)
        self.data['Future_Price'] = self.data['Close'].shift(-5)
        self.data['Buy_Signal_dnn'] = (self.data['Close'] < self.data['Future_Price']).astype(int)
        self.data['Sell_Signal_dnn'] = (self.data['Close'] > self.data['Future_Price']).astype(int)
        
        self.data.dropna(inplace=True)
        
        self.data.reset_index(drop=True, inplace=True)

    def prepare_data_for_dl(self, train_size = 0.9):
        """
        Prepares the data for deep learning models, creating training and test sets for buy and sell signals.
        """

        # Define the feature set X using price lags, volatility, returns, and spread
        features = ['Pt-1', 'Pt-2', 'Pt-3', 'Volatility', 'Returns', 'Spread', 'Buy_Signal_dnn', 'Sell_Signal_dnn']
        self.X = self.data[features]
        
        # Determine the cutoff for the test set
        cutoff = int(len(self.X) * (train_size))

        # Create a single DataFrame for the training set including both features and targets
        self.train_df = self.X.iloc[:cutoff]
        self.X_train_dnn = self.train_df.drop(['Buy_Signal_dnn', 'Sell_Signal_dnn'], errors='ignore', axis=1)
        self.Y_train_dnn_buy = self.train_df['Buy_Signal_dnn']
        self.Y_train_dnn_sell = self.train_df['Sell_Signal_dnn']

        # Create a single DataFrame for the test set including both features and targets
        self.test_df = self.X.iloc[cutoff:]
        self.X_test_dnn = self.test_df.drop(['Buy_Signal_dnn', 'Sell_Signal_dnn'], errors='ignore', axis=1)
        self.Y_test_dnn_buy = self.test_df['Buy_Signal_dnn']
        self.Y_test_dnn_sell = self.test_df['Sell_Signal_dnn']
        
    def build_and_train_dnn(self, direction = 'buy'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=50, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='softmax')
        ])

        metric = tf.keras.metrics.RootMeanSquaredError()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy function',
                      metrics=['accuracy'])
        
        # Train the best model on the full training dataset
      
        X_train = self.X_train_dnn.values
        y_train = self.Y_train_dnn_buy.values if direction == 'buy' else self.Y_train_dnn_sell.values
        X_test = self.X_test_dnn.values
        y_test = self.Y_test_dnn_buy.values if direction == 'buy' else self.Y_test_dnn_sell.values

        model_hist = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

        return model_hist
    
    def generate_predictions_dnn(self, model, direction = 'buy'):
        predictions = best_hist.predict(self.X.values)

        if direction == 'buy':
            self.data['Buy_Signal_dnn'] = predictions
        elif direction == 'sell':
            self.data['Sell_Signal_dnn'] = predictions   
            
    def build_and_train_lstm(self, direction='buy'):
        X_train = self.X_train_dnn.values.reshape((self.X_train_dnn.shape[0], 1, self.X_train_dnn.shape[1]))
        y_train = self.Y_train_dnn_buy.values if direction == 'buy' else self.Y_train_dnn_sell.values
        X_test = self.X_test_dnn.values.reshape((self.X_test_dnn.shape[0], 1, self.X_test_dnn.shape[1]))
        y_test = self.Y_test_dnn_buy.values if direction == 'buy' else self.Y_test_dnn_sell.values

        
        inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(50)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)

        return model

    def generate_predictions_lstm(self, model, direction='buy'):
        X = self.X.values.reshape((self.X.shape[0], 1, self.X.shape[1]))
        predictions = model.predict(X)

        if direction == 'buy':
            self.data['Buy_Signal_lstm'] = (predictions > 0.5).astype(int)
        elif direction == 'sell':
            self.data['Sell_Signal_lstm'] = (predictions < 0.5).astype(int)   
            
            
            
    def optimize_and_fit_models(self):
        self.prepare_data_for_dl()
        
        dnn_buy_model = self.build_and_train_dnn(direction = 'buy')
        self.generate_predictions_dnn(dnn_buy_model, direction = 'buy')
        dnn_sell_model = self.build_and_train_dnn(direction = 'sell')
        self.generate_predictions_dnn(dnn_buy_model, direction = 'sell')  
        
        lstm_buy_model = self.build_and_train_lstm(direction = 'buy')
        self.generate_predictions_lstm(lstm_buy_model, direction = 'buy')
        lstm_sell_model = self.build_and_train_lstm(direction = 'sell')
        self.generate_predictions_lstm(lstm_buy_model, direction = 'sell')
        
        
        
    def execute_trades(self, best = False, stop_loss=None, take_profit=None, n_shares=None):
        
        stop_loss = stop_loss or self.stop_loss
        take_profit = take_profit or self.take_profit
        n_shares = n_shares or self.n_shares
        
        if best == True:
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
                total_active_indicators = len(self.best_combination)
            
                    
        else: #False
            for indicator in self.active_indicators:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.active_indicators]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.active_indicators]].sum(axis=1)
                total_active_indicators = len(self.active_indicators)
        
        for i, row in self.data.iterrows():
            
           
            if total_active_indicators <= 2:
                if self.data.total_buy_signals.iloc[i] == total_active_indicators:
                    self._open_operation('long', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
                elif self.data.total_sell_signals.iloc[i] == total_active_indicators:
                    self._open_operation('short', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
            else:
                if self.data.total_buy_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares )
                elif self.data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
    
            self.check_close_operations(row, stop_loss, take_profit, n_shares)
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close'], n_shares) for op in self.operations if not op.closed) 
            self.strategy_value.append(total_value)
        

    def _open_operation(self, operation_type, row, stop_loss, take_profit, n_shares):
        if operation_type == 'long':
            stop_loss = row['Close'] * stop_loss
            take_profit = row['Close'] * take_profit
        else:  # 'short'
            stop_loss = row['Close'] * take_profit
            take_profit = row['Close'] * stop_loss

        self.operations.append(Operation(operation_type, row['Close'], row.name, n_shares, stop_loss, take_profit))
        if operation_type == 'long':
            self.cash -= row['Close'] * n_shares * (1 + self.com)
        else:  # 'short'
            self.cash += row['Close'] * n_shares * (1 - self.com)  # Incrementa el efectivo al abrir la venta en corto

    def check_close_operations(self, row,stop_loss, take_profit, n_shares):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= take_profit or row['Close'] <= stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= take_profit or row['Close'] >= stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * n_shares * (1 + self.com)  # Decrementa el efectivo al cerrar la venta en corto, basado en el nuevo precio
                   
                op.closed = True

    def calculate_operation_value(self, op, current_price, n_shares):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * n_shares if not op.closed else 0

    def plot_results(self, best = False):
        self.reset_strategy()
        if best == False:
            self.execute_trades()
        else:
            self.execute_trades(best=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()
        
    def run_combinations(self):
        all_indicators = ['dnn', 'lstm','cnn']
        for r in range(1, len(all_indicators) + 1):
            for combo in combinations(all_indicators, r):
                self.active_indicators = list(combo)
                print(f"Ejecutando con combinación de indicadores: {self.active_indicators}")
                self.execute_trades()
                
                final_value = self.strategy_value[-1]
                if final_value > self.best_value:
                    self.best_value = final_value
                    self.best_combination = self.active_indicators.copy()
                self.reset_strategy()

        print(f"Mejor combinación de indicadores: {self.best_combination} con un valor de estrategia de: {self.best_value}")

    def reset_strategy(self):
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]            
        
    def optimize_trade_parameters(self):
        def objective(trial):
            stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.90, 0.99)  
            take_profit_pct = trial.suggest_float('take_profit_pct', 1.01, 1.10) 
            n_shares = trial.suggest_int('n_shares', 1, 100) 

            self.reset_strategy()
            self.execute_trades(best=True, stop_loss=stop_loss_pct, take_profit=take_profit_pct, n_shares=n_shares)
            final_strategy_value = self.strategy_value[-1]

            return final_strategy_value

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)  # Ajustar el número de pruebas según sea necesario

        # Mejores parámetros encontrados
        best_params = study.best_params
        print(f"Mejores parámetros encontrados: {best_params}")

        # Aplicar los mejores parámetros a la estrategia
        self.stop_loss_pct = best_params['stop_loss_pct']
        self.take_profit_pct = best_params['take_profit_pct']
        self.n_shares = best_params['n_shares']        

    def test(self):
        test_file_mapping = {
            "5m": "aapl_5m_test.csv",
            "1h": "aapl_1h_test.csv",
            "1d": "aapl_1d_test.csv",
            "1m": "aapl_1m_test.csv"
        }
        self.load_data(self.file)
        self.calculate_indicators()
        self.define_buy_sell_signals()
        self.run_signals()
        self.execute_trades(best=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()        
    
 
     




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        