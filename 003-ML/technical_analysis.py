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
        
        return self.data
        
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
        
        
        #features_to_scale = ['Open', 'High', 'Low', 'Close', 'Returns', 'Volume_Trend',  'Volatility', 'Close_Trend', 'Spread']
        #scaler = RobustScaler()
        #self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale].fillna(0))
        self.data.dropna(inplace=True)
        
        self.data.reset_index(drop=True, inplace=True)
        
        
# Luis & Sofía

    def buy_signals(self):
        # Calcular el precio futuro utilizando un desplazamiento de 5 periodos
        self.data['Future_Price'] = self.data['Close'].shift(-5)
        # Definir señales de compra: 1 si el precio actual es menor que el precio futuro, 0 en caso contrario
        self.data['Buy_Signal_xgb'] = (self.data['Close'] < self.data['Future_Price']).astype(int)

    def sell_signals(self):
        # Utilizar el mismo precio futuro calculado para las señales de compra
        # Definir señales de venta: 1 si el precio actual es mayor que el precio futuro, 0 en caso contrario
        self.data['Sell_Signal_xgb'] = (self.data['Close'] > self.data['Future_Price']).astype(int)
        
    def prepare_data_for_ml(self, test_size = 0.2):
        """
        Prepares the data for machine learning models, creating training and test sets for buy and sell signals.
        """

        # Define the feature set X using price lags, volatility, returns, and spread
        features = ['Pt-1', 'Pt-2', 'Pt-3', 'Volatility', 'Returns', 'Spread']
        X = self.data[features]

        # Define the target variables y for buy and sell signals
        y_buy = self.data['Buy_Signal_xgb']
        y_sell = self.data['Sell_Signal_xgb']
        

        # Determine the cutoff for the test set
        cutoff = int(len(X) * (1 - test_size))

        # Create a single DataFrame for the training set including both features and targets
        self.train_df = self.data.iloc[:cutoff].copy()
        self.train_df['y_buy'] = y_buy.iloc[:cutoff]
        self.train_df['y_sell'] = y_sell.iloc[:cutoff]

        # Create a single DataFrame for the test set including both features and targets
        self.test_df = self.data.iloc[cutoff:].copy()
        self.test_df['y_buy'] = y_buy.iloc[cutoff:]
        self.test_df['y_sell'] = y_sell.iloc[cutoff:]


        
    def fit_xgboost(self, X_train, y_train, X_val, y_val, direction='buy'):
        
        """
            Train an XGBoost model and find the best hyperparameters.
        """
        
        def objective_xgb(trial):
            
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'max_leaves': trial.suggest_int('max_leaves', 0, 64),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            }
            model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='binary')
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_xgb, n_trials=25)  # Adjust the number of trials as necessary

        if direction == 'buy':
            self.best_xgbuy_params = study.best_params
        elif direction == 'sell':
            self.best_xgsell_params = study.best_params

        best_params = study.best_params
        best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        best_model.fit(X_train, y_train)

        # Generate predictions for the entire dataset
        X_total = self.data.drop(['Buy_Signal', 'Sell_Signal'], axis=1, errors='ignore')
        predictions = best_model.predict(X_total)

        if direction == 'buy':
            self.data['XGBoost_Buy_Signal'] = predictions
        elif direction == 'sell':
            self.data['XGBoost_Sell_Signal'] = predictions   


            
            
            
            
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
        study.optimize(objective, n_trials=3) 
        
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
            self.data['Logistic_Buy_Signal'] = predictions
        elif direction == 'sell':
            self.data['Logistic_Sell_Signal'] = predictions

        

    def optimize_and_fit_models(self):
        self.buy_model = self.fit_logistic_regression(self.X_vtrain, self.y_vtrain_buy, self.X_vtest, self.y_vtest_buy, direction='buy')
        self.sell_model = self.fit_logistic_regression(self.X_vtrain, self.y_vtrain_sell, self.X_vtest, self.y_vtest_sell, direction='sell')
        
 


    def execute_trades(self, best = False):
        
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
                    self._open_operation('long', row)
                elif self.data.total_sell_signals.iloc[i] == total_active_indicators:
                    self._open_operation('short', row)
            else:
                if self.data.total_buy_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row)
                elif self.data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row)
    
            # Verifica y cierra operaciones basadas en stop_loss o take_profit
            self.check_close_operations(row)
    
            # Actualiza el valor de la estrategia en cada iteración
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            #print(f"Fila: {i}, Valor de la estrategia: {total_value}")
            self.strategy_value.append(total_value)
        

    def _open_operation(self, operation_type, row):
        if operation_type == 'long':
            stop_loss = row['Close'] * 0.95
            take_profit = row['Close'] * 1.05
        else:  # 'short'
            stop_loss = row['Close'] * 1.05
            take_profit = row['Close'] * 0.95

        self.operations.append(Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))
        if operation_type == 'long':
            self.cash -= row['Close'] * self.n_shares * (1 + self.com)
        else:  # 'short'
            self.cash += row['Close'] * self.n_shares * (1 - self.com)  # Incrementa el efectivo al abrir la venta en corto
            
        #print(f"Operación {operation_type} iniciada en {row.name}, Precio: {row['Close']}, Cash restante: {self.cash}")

    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com)  # Decrementa el efectivo al cerrar la venta en corto, basado en el nuevo precio
                   
                op.closed = True
                #print(f"Operación {op.operation_type} cerrada en {row.name}, Precio: {row['Close']}, Cash resultante: {self.cash}")

    def calculate_operation_value(self, op, current_price):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0

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
        all_indicators = ['Logistic']
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
        

    def test(self):
        test_file_mapping = {
            "5m": "data/aapl_5m_test.csv",
            "1h": "data/aapl_1h_test.csv",
            "1d": "data/aapl_1d_test.csv",
            "1m": "data/aapl_1m_test.csv"
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
    
 
        
class MLModels(TradingStrategy):
    """
    A class to build and evaluate various machine learning models.
    Extends the functionality of the TradingStrategy class to include ML capabilities.
    """

    def __init__(self, file, k=5, threshold=0.01):
        """
        Initializes the MLModels instance.

        :param file: The path to the dataset file.
        :param k: The number of periods to look ahead for setting the target variable.
        :param threshold: The threshold for determining the buy/sell category.
        """
        super().__init__(file)
        self.k = k  # Number of periods forward to check the price
        self.threshold = threshold  # Threshold for determining the category
        
    def define_target_variable(self):
        """
        Defines the target variable based on future price movement.
        """
        self.data['Future Price'] = self.data['Close'].shift(-self.k)
        self.data['Target'] = (self.data['Future Price'] > self.data['Close'] * (1 + self.threshold)).astype(int)
        self.data.dropna(inplace=True)
    
    def split_data(self):
        """
        Splits the dataset into features (X) and the target variable (y), and then into training and testing sets.
        """
        X = self.data.drop(['Future Price', 'Target'], axis=1)
        y = self.data['Target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def xgboost(self):
        """
        Trains and evaluates an XGBoost model using the training data.
        """
        self.model = XGBClassifier(use_label_encoder=False)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print("XGBoost F1 Score:", f1_score(self.y_test, y_pred, average='binary'))
        print("\nXGBoost Classification Report:\n", classification_report(self.y_test, y_pred))
    
    # Placeholder for logistic regression model and the other model

    
    def run(self):
        """
        Executes the workflow for defining the target variable, splitting the data,
        training models, and evaluating their performance.
        """
        self.define_target_variable()
        self.split_data()
        self.xgboost()
        # Call other models here as needed, e.g., self.logistic_regression()





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        