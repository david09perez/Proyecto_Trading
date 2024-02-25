import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna

class Operation:
    
    """
    Represents a trading operation based on techical analysis indicators, specifically
    tailored for executing strategies with AAPL (Apple Inc.) closing prices over a given period.
    This class is designed to facilitate the tracking and management of trades, incorporating
    risk management parameters such as stop loss and take profit levels.
    
    """
    
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        
        """
        Initializes an instance of the Operation class, setting up the trade details based on the closing prices of AAPL and the chosen 
        technical analysis strategy.

        Parameters:
            operation_type: 'buy' for entering a long position or 'sell' for shorting, based on the technical indicator signals.
            bought_at: The AAPL closing price at which the trade is opened.
            timestamp: The moment the trade is executed, providing a temporal reference for the operation.
            n_shares: The volume of the trade, in terms of how many shares of AAPL are being traded.
            stop_loss: The set price level to exit the trade to minimize losses if the market moves unfavorably
            take_profit: The set price level to exit the trade when the desired price objective is achieved
            
        """
        
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
        
        """
        Initializes the trading strategy, setting up initial parameters, loading data, calculating indicators, and preparing 
        for trading execution.
        
         Parameters:
            file: The identifier for the time frame of the data to be used in the strategy.
        
        """
        
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
        self.indicators = {}
        self.active_indicators = []
        self.calculate_indicators()
        self.define_buy_sell_signals()
        self.run_signals()
        self.best_combination = None
        self.optimize_combination = None
        self.best_value = 0
        
    def load_data(self, time_frame):
        
        """
        Loads historical stock data from the CSV file corresponding to the specified time frame. This function
        ensures that the data is available and in the correct format for analysis. It raises an error if the
        time frame is unsupported, ensuring that the strategy operates on valid data sets.
        
        Parameters:
            time_frame: A string identifier for the desired time frame of the stock data.
            
        """
        
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def calculate_indicators(self):
        
        """
        Calculates a set of technical indicators on the loaded stock data, enriching the data set with additional
        features for analysis. These indicators include but are not limited to RSI (Relative Strength Index), SMA
        (Simple Moving Averages), MACD (Moving Average Convergence Divergence), SAR (Parabolic SAR), ADX (Average
        Directional Index), and Stochastic Oscillator. Each indicator provides different insights into the market's
        behavior and is used to generate trading signals.
        
        """ 
        
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=14)
        self.data['RSI'] = rsi_indicator.rsi()

        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=5)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=21)
        self.data['SHORT_SMA'] = short_ma.sma_indicator()
        self.data['LONG_SMA'] = long_ma.sma_indicator()

        macd = ta.trend.MACD(close=self.data['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()
        
        self.data['SAR'] = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close']).psar()
        
        adx_indicator = ta.trend.ADXIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=14)
        self.data['ADX'] = adx_indicator.adx()
        self.data['+DI'] = adx_indicator.adx_pos()
        self.data['-DI'] = adx_indicator.adx_neg()
        
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=14, smooth_window=3)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()
        
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True) 

    def define_buy_sell_signals(self):
        
        """
        Defines the logic for generating buy and sell signals for each technical indicator used in the strategy.
        This function populates a dictionary mapping each indicator to its corresponding buy and sell functions,
        facilitating the dynamic execution of these signals based on the current market data. This setup allows
        for a modular and flexible approach to testing different indicators and their combinations.
        
        """
        
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'SAR' : {'buy': self.sar_buy_signal, 'sell': self.sar_sell_signal},
            'ADX' : {'buy': self.adx_buy_signal, 'sell': self.adx_sell_signal}, 
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal}
        }

    def activate_indicator(self, indicator_name):
        
        """
        Activates a given indicator by adding it to the list of active indicators. This function allows the strategy
        to selectively include indicators in the analysis, enabling a dynamic approach to evaluating the effectiveness
        of various technical indicators in real-time trading scenarios.
        
        Parameters:
            indicator_name: The name of the indicator to be activated for the strategy.
            
        """
        
        if indicator_name in self.indicators:
                self.active_indicators.append(indicator_name)
                         
    def stoch_buy_signal(self, row, prev_row=None):
        
        """
        Determines a buy signal based on the Stochastic Oscillator. 
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and prev_row['stoch_%K'] < prev_row['stoch_%D'] and row['stoch_%K'] > row['stoch_%D'] and row['stoch_%K'] < 20
    
    def stoch_sell_signal(self, row, prev_row=None):
        
        """
        Determines a sell signal based on the Stochastic Oscillator.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and prev_row['stoch_%K'] > prev_row['stoch_%D'] and row['stoch_%K'] < row['stoch_%D'] and row['stoch_%K'] > 80
    
    def rsi_buy_signal(self, row, prev_row=None):
        
        """
        Generates a buy signal when the RSI falls below a certain threshold, indicating that the stock may be oversold and could 
        be poised for a price increase. This function looks at the RSI value of the current row to decide on the signal.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data, not used in this function.
            
        """
        
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        
        """
        Generates a sell signal when the RSI rises above a certain threshold, indicating that the stock may be overbought and could 
        be poised for a price decrease. This function evaluates the RSI value of the current row to generate the signal.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data, not used in this function.
            
        """
        
        return row.RSI > 70

    def sma_buy_signal(self, row, prev_row=None):
        
        """
        Generates a buy signal based on SMA crossovers. A buy signal occurs when the short-term SMA crosses above the long-term 
        SMA, suggesting an upward price momentum and a potential bullish trend. This function requires both the current and 
        previous rows to identify the crossover point.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and prev_row['LONG_SMA'] > prev_row['SHORT_SMA'] and row['LONG_SMA'] < row['SHORT_SMA']

    def sma_sell_signal(self, row, prev_row=None):
        
        """
        Generates a sell signal based on SMA crossovers. A sell signal occurs when the short-term SMA crosses below the long-term
        SMA, suggesting a downward price momentum and a potential bearish trend. This function relies on both the current and 
        previous rows to detect the crossover event.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and prev_row['LONG_SMA'] < prev_row['SHORT_SMA'] and row['LONG_SMA'] > row['SHORT_SMA']

    def macd_buy_signal(self, row, prev_row=None):
        
        """
        Generates a buy signal when the MACD line crosses above the signal line, indicating a potential bullish reversal
        and upward price momentum. This function compares the MACD values between the current and previous rows to
        detect the crossover.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
        return False

    def macd_sell_signal(self, row, prev_row=None):
        
        """
        Generates a sell signal when the MACD line crosses below the signal line, indicating a potential bearish reversal
        and downward price momentum. This function uses the MACD values from the current and previous rows to identify
        the crossover point.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
        return False

    def sar_buy_signal(self, row, prev_row=None):
        
        """
        Generates a buy signal based on the Parabolic SAR indicator. A buy signal is generated when the SAR value is
        below the current price, indicating a potential uptrend. The function checks the SAR values against the close
        prices of the current and previous rows to determine the signal.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and row['SAR'] < row['Close'] and prev_row['SAR'] > prev_row['Close']

    def sar_sell_signal(self, row, prev_row=None):
        
        """
        Generates a sell signal based on the Parabolic SAR indicator. A sell signal is generated when the SAR value is
        above the current price, indicating a potential downtrend. The function evaluates the SAR values against the
        close prices of the current and previous rows to generate the signal.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and row['SAR'] > row['Close'] and prev_row['SAR'] < prev_row['Close']

    def adx_buy_signal(self, row, prev_row=None):
        
        """
        Generates a buy signal when the ADX indicates a strong upward trend, which is when the +DI line crosses above
        the -DI line and the ADX value is above a certain threshold, suggesting a strengthening trend. The
        function requires both the current and previous rows to identify the crossover and confirm the trend strength.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and row['+DI'] > row['-DI'] and row['ADX'] > 25 and prev_row['+DI'] < prev_row['-DI']

    def adx_sell_signal(self, row, prev_row=None):
        
        """
        Generates a sell signal when the ADX indicates a strong downward trend, which is when the +DI line crosses below
        the -DI line and the ADX value is above a certain threshold, indicating a strong bearish trend. The
        function uses data from both the current and previous rows to detect the crossover and assess the trend's strength.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return prev_row is not None and row['+DI'] < row['-DI'] and row['ADX'] > 25 and prev_row['+DI'] > prev_row['-DI'] 
        
    def run_signals(self):
        
        """
        Iterates through each technical indicator used in the strategy and applies its buy and sell signal functions
        to the entire dataset. This method dynamically generates columns in the data DataFrame to mark where each
        indicator suggests a buy or sell action. The signals are initially boolean and are then converted to integers 
        for easier quantitative analysis.
        
        """
        
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data.apply(lambda row: self.indicators[indicator]['buy'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1)
            self.data[indicator + '_sell_signal'] = self.data.apply(lambda row: self.indicators[indicator]['sell'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1)
    
        # Ensure that buy and sell signals are converted to numeric values (1 for True, 0 for False)
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)
        
    def execute_trades(self, best = False):
        
        """
        Executes trades based on the aggregated buy and sell signals from either all active indicators or the best
        combination of indicators found through optimization. The strategy considers the total number of buy or sell
        signals for each row in the dataset and decides whether to open a long or short position based on predefined
        conditions, such as whether the total signals match the number of active indicators or exceed half of them.
        If no actives indicators when runned, it will take all available indicators
        
        Parameters:
            best: If True, the strategy uses the best combination of indicators for trade execution.
                  If False, it uses all currently active indicators.
                         
        """
        
        if best == True:
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
                total_active_indicators = len(self.best_combination)
            
                    
        else: #False
            if len(self.active_indicators) == 0:
                self.active_indicators = list(self.indicators.keys())
            else:
                 self.active_indicators = self.active_indicators 
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
    
            # Check and close operations based on stop_loss or take_profit
            self.check_close_operations(row)
    
            # Update the value of the strategy in each iteration
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            #print(f"Fila: {i}, Valor de la estrategia: {total_value}")
            self.strategy_value.append(total_value)
        

    def _open_operation(self, operation_type, row):
        
        """
        Opens a new trading operation based on the given operation type, long or short, and the market data at
        the current row. It sets the stop loss and take profit levels based on the current price and updates the cash
        balance to reflect the cost of the trade. The new operation is then added to the list of ongoing operations.
        
        Parameters:
            operation_type: The type of operation to open, long or short.
            row: The current row of market data used to determine the trade's entry price.
            
        """
        
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
            self.cash += row['Close'] * self.n_shares * (1 - self.com)  # Increase cash when opening a short sale
            
        #print(f"Operación {operation_type} iniciada en {row.name}, Precio: {row['Close']}, Cash restante: {self.cash}")

    def check_close_operations(self, row):
        
        """
        Iterates through all open operations to check if any should be closed based on the current market data. An
        operation is closed if the current price reaches the stop loss or take profit levels. The cash balance is
        updated to reflect the closing of the operation, and the operation's status is marked as closed.
        
        Parameters:
            row: The current row of market data used to check for stop loss or take profit conditions.
            
        """
        
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com)  # Decrease cash when closing a short sale, based on the new price
                   
                op.closed = True
                #print(f"Operación {op.operation_type} cerrada en {row.name}, Precio: {row['Close']}, Cash resultante: {self.cash}")

    def calculate_operation_value(self, op, current_price):
        
        """
        Calculates the current value of an open operation based on the difference between the current price and the
        operation's entry price, multiplied by the number of shares. The value is positive for profitable operations
        and negative for losing ones. Closed operations have a value of zero.
        
        Parameters:
            op: The operation for which to calculate the current value.
            current_price: The current market price of the stock.
            
        """
        
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0

    def plot_results(self, best = False):
        
        """
        Resets the strategy and executes trades using the best combination of indicators found through optimization.
        It then plots the strategy's performance over time, showing how the strategy's value changes with each trade.
        This method is useful for visually assessing the effectiveness of the best indicator combination.
        
        """
        
        self.reset_strategy()
        if best ==  True:
            self.execute_trades(best=True)
        else:
            self.execute_trades()
            
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()
        
    def run_combinations(self):
        
        """
        Tests all possible combinations of the available technical indicators to find the combination that yields the
        best performance. It iterates through each combination, executes trades based on the signals from the
        indicators in the combination, and tracks the final value of the strategy. The combination with the highest
        final value is deemed the best.
        
        """
        
        all_indicators = list(self.indicators.keys())
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
        
        """
        Resets the strategy to its initial state by clearing all ongoing operations, resetting the cash balance to its
        original amount, and setting the strategy value back to its starting point.
        
        """
        
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]        
        
   
    def optimize_parameters(self, prior = False):
        
        """
        Uses an optimization framework Optuna to find the best parameters for each indicator in the best
        combination of indicators. It defines an objective function that runs the strategy with varying indicator
        parameters and seeks to maximize the strategy's final value. The best parameters found are then applied to
        the indicators for future strategy execution.
        
        """
        
        def objective(trial):
            
            """
            The objective function used by the optimization framework Optuna to find the optimal parameters for each
            technical indicator within the best combination. This function is called each time with a different set of 
            parameters proposed by the optimizer. 

            It resets the strategy, sets new parameters for each indicator, executes the strategy, and returns the final 
            value of the strategy.

            Parameters:
                trial: An object representing a single call of the objective function, containing methods to propose
                       new parameter values.

            """
            
            if prior ==True:
                self.optimize_combination = self.active_indicators
                
            else:
                self.optimize_combination = self.best_combination
                
            self.reset_strategy()
            # Set the parameters for each active indicator in the best combination
            for indicator in self.optimize_combination:
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 5, 30)
                    self.set_rsi_parameters(rsi_window)
                elif indicator == 'SMA':
                    short_ma_window = trial.suggest_int('short_ma_window', 5, 20)
                    long_ma_window = trial.suggest_int('long_ma_window', 21, 50)
                    self.set_sma_parameters(short_ma_window, long_ma_window)
                elif indicator == 'MACD':
                    macd_fast = trial.suggest_int('macd_fast', 10, 20)
                    macd_slow = trial.suggest_int('macd_slow', 21, 40)
                    macd_sign = trial.suggest_int('macd_sign', 5, 15)
                    self.set_macd_parameters(macd_fast, macd_slow, macd_sign)
                elif indicator == 'SAR':
                    sar_step = trial.suggest_float('sar_step', 0.01, 0.1)
                    sar_max_step = trial.suggest_float('sar_max_step', 0.1, 0.5)
                    self.set_sar_parameters(sar_step, sar_max_step)
                
                elif indicator == 'ADX':
                    adx_window = trial.suggest_int('adx_window', 10, 30)
                    self.set_adx_parameters(adx_window)
                    
                if indicator == 'Stoch':
                    stoch_k_window = trial.suggest_int('stoch_k_window', 5, 21) 
                    stoch_d_window = trial.suggest_int('stoch_d_window', 3, 14)  
                    stoch_smoothing = trial.suggest_int('stoch_smoothing', 3, 14)  
                
                    self.set_stoch_parameters(stoch_k_window, stoch_d_window, stoch_smoothing)
                    
            take_profit_multiplier = trial.suggest_float('take_profit', 1.01, 1.2)
            stop_loss_multiplier = trial.suggest_float('stop_loss', 0.87, 0.99)
            number_of_shares = trial.suggest_int('n_shares', 1, 500)
    
            self.take_profit_multiplier = take_profit_multiplier
            self.stop_loss_multiplier = stop_loss_multiplier
            self.n_shares = number_of_shares
            
            self.run_signals()
            
            if prior==True:
                self.execute_trades()
            else:
                self.execute_trades(best= True)
            #print(len(self.strategy_value))
   
            return self.strategy_value[-1]
    
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)  
    
        # Print and apply the best parameters found for each indicator
        print(f"Mejores parámetros encontrados: {study.best_params}")
        for indicator in self.optimize_combination:
            
            if indicator == 'RSI':
                self.set_rsi_parameters(study.best_params['rsi_window'])
            elif indicator == 'SMA':
                self.set_sma_parameters(study.best_params['short_ma_window'], study.best_params['long_ma_window'])
            elif indicator == 'MACD':
                self.set_macd_parameters(study.best_params['macd_fast'], study.best_params['macd_slow'], study.best_params['macd_sign'])
            elif indicator == 'SAR':
                self.set_sar_parameters(study.best_params['sar_step'], study.best_params['sar_max_step'])
            elif indicator == 'ADX':
                self.set_adx_parameters(study.best_params['adx_window'])
            elif indicator == 'Stoch':
                self.set_stoch_parameters(study.best_params['stoch_k_window'], study.best_params['stoch_d_window'], study.best_params['stoch_smoothing'])
                
        self.take_profit_multiplier = study.best_params['take_profit']
        self.stop_loss_multiplier = study.best_params['stop_loss']
        self.n_shares = study.best_params['n_shares']
                    
            
    def set_rsi_parameters(self, window):
        
        """
        Sets the parameters for the RSI indicator, specifically the look-back period.
        This method recalculates the RSI values for the entire dataset based on the new window size.

        Parameters:
            window: The look-back period to calculate the RSI.

        """
        
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window)
        self.data['RSI'] = rsi_indicator.rsi()
    
    def set_sma_parameters(self, short_window, long_window):
        
        """
        Sets the parameters for calculating the short-term and long-term Simple Moving Averages. This method
        recalculates the SMA values for the entire dataset based on the new window sizes for both averages.

        Parameters:
            short_window: The look-back period for the short-term SMA, typically a smaller number.
            long_window: The look-back period for the long-term SMA, typically a larger number than short_window.

        """
        
        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=short_window)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=long_window)
        self.data['SHORT_SMA'] = short_ma.sma_indicator()
        self.data['LONG_SMA'] = long_ma.sma_indicator()
    
    def set_macd_parameters(self, fast, slow, sign):
        
        """
        Sets the parameters for the MACD (Moving Average Convergence Divergence) indicator, including the fast period,
        slow period, and signal line smoothing period. This method recalculates the MACD and its signal line for the
        entire dataset based on the new parameters.

        Parameters:
            fast: The look-back period for the fast moving average.
            slow: The look-back period for the slow moving average.
            sig: The look-back period for the signal line, which is a smoothed version of the MACD.

        """
        
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=sign)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()            
    
    def set_sar_parameters(self, step, max_step):
        
        """
        Sets the parameters for the Parabolic Stop and Reverse indicator, including the step and the maximum step. 
        This method recalculates the SAR values for the entire dataset based on the new parameters.

        Parameters:
            step: The acceleration factor for the SAR calculation.
            max_step: The maximum value the step can reach.

        """
        
        sar_indicator = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], step=step, max_step=max_step)
        self.data['SAR'] = sar_indicator.psar()
    
    def set_adx_parameters(self, window):
        
        """
        Sets the parameters for the Average Directional Index indicator, specifically the look-back period for 
        calculating the indicator. This method recalculates the ADX, +DI, and -DI for the entire dataset based 
        on the new window size.

        Parameters:
            window: The look-back period for calculating the ADX and its directional indicators.

        """
        
        adx_indicator = ta.trend.ADXIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=window)
        self.data['ADX'] = adx_indicator.adx()
        self.data['+DI'] = adx_indicator.adx_pos()
        self.data['-DI'] = adx_indicator.adx_neg()
    
    def set_stoch_parameters(self, k_window, d_window, smoothing):
        
        """
        Sets the parameters for the Stochastic Oscillator indicator. This method recalculates the Stochastic 
        Oscillator values for the entire dataset based on the new parameters.

        Parameters:
            k_window: The look-back period for calculating %K.
            d_window: The smoothing period for %K, affecting the smoothness of the %K line.
            smoothing: The look-back period for calculating %D, which is a moving average of %K.

        """
        
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=k_window, smooth_window=d_window)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal().rolling(window=smoothing).mean()
                    
    def test(self):
        
        """
        Tests the trading strategy on a separate test dataset to evaluate its performance on unseen data. This method
        aims to simulate real-world application of the strategy and assess its robustness and potential profitability.
        
        It loads the test data, recalculates the indicators, runs the signals, and executes the trades based on the
        best combination of indicators and parameters previously determined. Finally, it plots the strategy's
        performance over the test period.

        """
        self.reset_strategy()
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
    
    
    def show_ADX_strat(self):
        
        """
        Visualizes the ADX strategy's performance along with the closing prices and the ADX indicator values. This method
        plots the stock's closing prices in the first subplot and the ADX, +DI, and -DI indicators in the second subplot,
        providing a graphical representation of how the ADX strategy interacts with price movements and trend strength
        over a selected period. It's particularly useful for analyzing the effectiveness and timing of the ADX-based
        trading signals.

        """
        
        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.data.Close.iloc[:214], label='Close Price')
        ax1.set_title('Closing Prices and ADX Indicator')
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(self.data['ADX'].iloc[:214], label='ADX', color='black')
        ax2.plot(self.data['+DI'].iloc[:214], label='+DI', color='green')
        ax2.plot(self.data['-DI'].iloc[:214], label='-DI', color='red')

        ax2.axhline(25, color='gray', linestyle='--', label = 'Trend Strength Threshold')

        ax2.set_title('ADX and Directional Indicators')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def show_RSI(self):
        
        """
        Visualizes the Relative Strength Index alongside the closing prices of the stock. This function plots the 
        closing prices in the upper subplot and the RSI values in the lower subplot, with horizontal lines marking
        the typical overbought and oversold thresholds. This visualization helps in identifying potential buy or 
        sell signals based on RSI values crossing these thresholds.
        
        """
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        axs[0].plot(self.data.Close[:214])
        axs[0].set_title('Closing Prices')

        axs[1].plot(self.data.RSI[:214])
        axs[1].plot([0, 214], [70, 70], label="Upper Threshold")
        axs[1].plot([0, 214], [30, 30], label="Lower Threshold")
        axs[1].set_title('RSI')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

        
    def show_SMAs(self):
        
        """
        Displays the short-term and long-term Simple Moving Averages along with the closing prices. This function plots 
        the price and the two SMAs over the same period, allowing for a visual comparison and identification of potential 
        crossover points, which are commonly used as buy or sell signals in moving average-based trading strategies.

        """
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.Close[:250], label='price')
        plt.plot(self.data.SHORT_SMA[:250], label='SMA(5)')
        plt.plot(self.data.LONG_SMA[:250], label='SMA(21)')
        plt.legend()
        plt.show()



    def show_MACD(self):
        
        """
        Visualizes the Moving Average Convergence Divergence line and its signal line over a specified period,
        followed by a separate plot showing the histogram. The first plot helps in identifying crossover points
        between the MACD line and the signal line, which are used as buy or sell signals. The second plot, featuring
        the MACD histogram, provides a visual representation of the momentum by highlighting the difference between
        the MACD line and the signal line.

        """
        
        plt.figure(figsize=(12, 8))

        # Plot the MACD
        plt.plot(self.data.index[:214], self.data['MACD'][:214], label='MACD', color='blue')

        # Plot the signal line
        plt.plot(self.data.index[:214], self.data['Signal_Line'][:214], label='Signal Line', color='red')

        # Add title and legend
        plt.title('MACD and Signal Line')
        plt.legend()

        plt.show()

        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']

        plt.figure(figsize=(12, 6))

        # Plot the MACD and the signal line
        plt.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data.index, self.data['Signal_Line'], label='Signal Line', color='red')

        # Fill the histogram between MACD and the signal line
        # We will use a different color depending on whether the histogram is positive or negative
        plt.bar(self.data.index, self.data['MACD_Histogram'], label='MACD Histogram', color=['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']])

        # Add title and legend
        plt.title('MACD, Signal Line, and Histogram')
        plt.legend()
        plt.xlim(100, 200)
        plt.ylim(-1, 1)
        plt.show()
        
        
        
    def show_SAR(self):
        
        """
        Plots the Parabolic Stop and Reverse indicator points on top of the closing price chart. The SAR points
        are colored differently to indicate potential buy or sell signals. This visualization helps in understanding 
        the SAR indicator's signals in the context of price movements, providing insights into potential trend reversals.

        """
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.data.Close.iloc[:214], label = 'Close Price')

        legend_added_buy = False
        legend_added_sell = False

        for i in range(214):
            if self.data.SAR.iloc[i] < self.data.Close.iloc[i]:
                color = 'green'
                label = 'Buy Order' if not legend_added_buy else None  
                legend_added_buy = True
            else:
                color = 'red'
                label = 'Sell Order' if not legend_added_sell else None  
                legend_added_sell = True

            plt.scatter(self.data.index[i], self.data.SAR.iloc[i], color=color, s=20, label=label)

        if not legend_added_buy:
            plt.scatter([], [], color='green', s=20, label='Buy Order')
        if not legend_added_sell:
            plt.scatter([], [], color='red', s=20, label='Sell Order')

        plt.title('SAR Indicator with Closing Prices')
        plt.legend()
        plt.show()
        
    def plot_stochastic_signals(self):
        
        """
        Plots the Stochastic Oscillator buy and sell signals on the price chart. Buy signals are marked with green
        upward-pointing triangles at points where the %K line crosses above the %D line from below, and sell signals
        are marked with red downward-pointing triangles at points where the %K line crosses below the %D line from above.
        This function helps in visually identifying the timing and positioning of these stochastic-based trading signals
        relative to the price action.

        """
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index[:250], self.data['Close'][:250], label='Close Price')

        # Buy signals
        plt.scatter(self.data.index[:250][self.data['Stoch_buy_signal'][:250] == 1], self.data['Close'][:250][self.data['Stoch_buy_signal'][:250] == 1], color='green', marker='^', label='Buy Signal')
        # Sell signals
        plt.scatter(self.data.index[:250][self.data['Stoch_sell_signal'][:250] == 1], self.data['Close'][:250][self.data['Stoch_sell_signal'][:250] == 1], color='red', marker='v', label='Sell Signal')

        plt.title('Stochastic Buy/Sell Signals')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.legend()
        plt.show()











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        