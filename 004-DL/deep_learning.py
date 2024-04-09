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
            "5m": "aapl_5m_train.csv",
            "1h": "aapl_1h_train.csv",
            "1d": "aapl_1d_train.csv",
            "1m": "aapl_1m_train.csv"
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
        
        
# Luis Robles
   
    def prepare_data_for_dl(self, train_size = 0.8):
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
        
    def build_and_train_dnn(self, X_train, X_test, y_train, y_test):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=50, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
        ])

        metric = tf.keras.metrics.RootMeanSquaredError()
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=[metric])

        model_hist = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

        return model, model_hist
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_dnn, n_trials=1)  # Adjust the number of trials as necessary

        # Store the best parameters
        if direction == 'buy':
            self.best_dnn_buy_params = study.best_params
        elif direction == 'sell':
            self.best_dnn_sell_params = study.best_params

        # Train the best model on the full training dataset
        best_params = study.best_params
        X_train = self.X_train_dnn.values
        y_train = self.Y_train_dnn_buy.values if direction == 'buy' else self.Y_train_dnn_sell.values
        X_test = self.X_test_dnn.values
        y_test = self.Y_test_dnn_buy.values if direction == 'buy' else self.Y_test_dnn_sell.values

        best_model, _ = self.build_and_train_dnn(X_train, X_test, y_train, y_test)

        # Generate predictions for the entire dataset
        predictions = best_model.predict(self.X.values)

        # Add predictions back to the dataset
        if direction == 'buy':
            self.data['DNN_buy_signal'] = predictions
        elif direction == 'sell':
            self.data['DNN_sell_signal'] = predictions  
    
    #Zata y DArio    
    
    def fit_svm(self, direction='buy'):
        """
        Train an SVM model and find the best hyperparameters.
        """
        X_train = self.X_train_xgb
        y_train = self.Y_train_xgb_buy if direction == 'buy' else self.Y_train_xgb_sell
        X_val = self.X_test_xgb
        y_val = self.Y_test_xgb_buy if direction == 'buy' else self.Y_test_xgb_sell

        def objective_svm(trial):
            
            C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            gamma = 'scale' if kernel == 'linear' else trial.suggest_float('gamma', 1e-6, 1e+1, log=True)
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='binary')

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_svm, n_trials=1)  # Adjust the number of trials as necessary

        # Store the best parameters
        if direction == 'buy':
            self.best_svmbuy_params = study.best_params
        elif direction == 'sell':
            self.best_svmsell_params = study.best_params

        # Train the best model on the full training dataset
        best_params = study.best_params
        best_model = SVC(**best_params)
        best_model.fit(X_train, y_train)
        
        

        # Generate predictions for the entire dataset
        X_total = self.X.drop(['Buy_Signal_xgb', 'Sell_Signal_xgb'], axis=1, errors='ignore')
        predictions = best_model.predict(X_total)

        # Add predictions back to the dataset
        if direction == 'buy':
            self.data['SVM_buy_signal'] = predictions
        elif direction == 'sell':
            self.data['SVM_sell_signal'] = predictions

         
            
    def prepare_data_for_log_model(self):
        relevant_columns = ['Returns', 'Volatility', 'Close_Trend','Volume_Trend', 'Spread',  'LR_Buy_Signal', 'LR_Sell_Signal'] #,'RSI_buy_signal','Volume_Trend',
       #'RSI_sell_signal', 'SMA_buy_signal', 'SMA_sell_signal','MACD_buy_signal', 'MACD_sell_signal', 'SAR_buy_signal',
       #'SAR_sell_signal', 'ADX_buy_signal', 'ADX_sell_signal' ,'Spread','Open', 'High', 'Low', 'Close', ]
        self.processed_data = self.data[relevant_columns]
        split_idx = int(len(self.processed_data) * 0.75)
        
        self.vtrain_data = self.processed_data.iloc[:split_idx]
        self.X_vtrain = self.vtrain_data.drop(['LR_Buy_Signal', 'LR_Sell_Signal'], errors='ignore', axis=1)
        self.y_vtrain_buy = self.vtrain_data['LR_Buy_Signal']
        self.y_vtrain_sell = self.vtrain_data['LR_Sell_Signal']
        
        self.vtest_data = self.processed_data.iloc[split_idx:]
        self.X_vtest = self.vtest_data.drop(['LR_Buy_Signal', 'LR_Sell_Signal'], errors='ignore', axis=1)
        self.y_vtest_buy = self.vtest_data['LR_Buy_Signal']
        self.y_vtest_sell = self.vtest_data['LR_Sell_Signal']
          
            
    def fit_logistic_regression(self, X_train, y_train, X_val, y_val, direction='buy'):

        def objective(trial):
            C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            
            model = LogisticRegression(C=C, fit_intercept=fit_intercept, penalty='elasticnet', l1_ratio=l1_ratio, solver='saga', max_iter=10000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='binary')
            
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1) 
        
        if direction == 'buy':
            self.best_buylog_params = study.best_params
        elif direction == 'sell':
            self.best_selllog_params = study.best_params
        
        best_log_params = study.best_params

        best_model = LogisticRegression(**best_log_params, penalty='elasticnet', solver='saga', max_iter=10_000)
        best_model.fit(X_train, y_train)
        signal_columns = ['LR_Buy_Signal', 'LR_Sell_Signal', 'Logistic_Buy_Signal', 'Logistic_Sell_Signal']
        X_total = self.processed_data.drop(signal_columns, axis=1, errors='ignore') 
        
        predictions = best_model.predict(X_total)

        if direction == 'buy':
            self.data['Logistic_buy_signal'] = predictions
        elif direction == 'sell':
            self.data['Logistic_sell_signal'] = predictions

        

    def optimize_and_fit_models(self):
        self.prepare_data_for_ml()
        self.prepare_data_for_log_model()
        
        self.fit_logistic_regression(self.X_vtrain, self.y_vtrain_buy, self.X_vtest, self.y_vtest_buy, direction='buy')
        self.fit_logistic_regression(self.X_vtrain, self.y_vtrain_sell, self.X_vtest, self.y_vtest_sell, direction='sell')
        
        self.fit_xgboost(direction = 'buy')
        self.fit_xgboost(direction = 'sell')
        
        self.fit_svm(direction = 'buy')
        self.fit_svm(direction = 'sell')
 

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
    
            # Verifica y cierra operaciones basadas en stop_loss o take_profit
            self.check_close_operations(row, stop_loss, take_profit, n_shares)
    
            # Actualiza el valor de la estrategia en cada iteración
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close'], n_shares) for op in self.operations if not op.closed)
            #print(f"Fila: {i}, Valor de la estrategia: {total_value}")
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
            
        #print(f"Operación {operation_type} iniciada en {row.name}, Precio: {row['Close']}, Cash restante: {self.cash}")

    def check_close_operations(self, row,stop_loss, take_profit, n_shares):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= take_profit or row['Close'] <= stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= take_profit or row['Close'] >= stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * n_shares * (1 + self.com)  # Decrementa el efectivo al cerrar la venta en corto, basado en el nuevo precio
                   
                op.closed = True
                #print(f"Operación {op.operation_type} cerrada en {row.name}, Precio: {row['Close']}, Cash resultante: {self.cash}")

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
        all_indicators = ['Logistic', 'XGBoost','SVM']
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
    
 
     




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        