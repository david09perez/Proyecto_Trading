import pandas as pd
import matplotlib.pyplot as plt
import ta


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
    def __init__(self):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.com = 0.00125
        self.strategy_value = [1_000_000]
        self.n_shares = 10
        self.file_mapping = {
            "5m": "data/aapl_5m_train.csv",
            "1h": "data/aapl_1h_train.csv",
            "1d": "data/aapl_1d_train.csv",
            "1m": "data/aapl_1m_train.csv"
        }
        self.indicators = {}
        self.active_indicators = []

    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)


    def calculate_indicators(self):
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
        
        self.data.dropna(inplace=True)


    def define_buy_sell_signals(self):
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'SAR' : {'buy': self.sar_buy_signal, 'sell': self.sar_sell_signal},
            'ADX' : {'buy': self.adx_buy_signal, 'sell': self.adx_sell_signal}
        }

    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)


    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70

    def sma_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['LONG_SMA'] > prev_row['SHORT_SMA'] and row['LONG_SMA'] < row['SHORT_SMA']

    def sma_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['LONG_SMA'] < prev_row['SHORT_SMA'] and row['LONG_SMA'] > row['SHORT_SMA']

    def macd_buy_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
        return False

    def macd_sell_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
        return False

    def sar_buy_signal(self, row, prev_row=None):
        return prev_row is not None and row['SAR'] < row['Close'] and prev_row['SAR'] > prev_row['Close']

    def sar_sell_signal(self, row, prev_row=None):
        return prev_row is not None and row['SAR'] > row['Close'] and prev_row['SAR'] < prev_row['Close']

    def adx_buy_signal(self, row, prev_row=None):
        return prev_row is not None and row['+DI'] > row['-DI'] and row['ADX'] > 25 and prev_row['+DI'] < prev_row['-DI']

    def adx_sell_signal(self, row, prev_row=None):
        return prev_row is not None and row['+DI'] < row['-DI'] and row['ADX'] > 25 and prev_row['+DI'] > prev_row['-DI']    
        
    def execute_trades(self):
        if 'RSI' not in self.data.columns:
            self.calculate_indicators()
    
        # Añadir filas previas desplazadas para las señales que requieran prev_row
        self.data['prev_Close'] = self.data['Close'].shift(1)
        self.data['prev_LONG_SMA'] = self.data['LONG_SMA'].shift(1)
        self.data['prev_SHORT_SMA'] = self.data['SHORT_SMA'].shift(1)
        self.data['prev_MACD'] = self.data['MACD'].shift(1)
        self.data['prev_Signal_Line'] = self.data['Signal_Line'].shift(1)
        self.data['prev_SAR'] = self.data['SAR'].shift(1)
        self.data['prev_+DI'] = self.data['+DI'].shift(1)
        self.data['prev_-DI'] = self.data['-DI'].shift(1)
    
        # Calcular señales de compra y venta para cada indicador y almacenarlas en el DataFrame
        self.data['RSI_buy'] = self.data['RSI'] < 30
        self.data['RSI_sell'] = self.data['RSI'] > 70
        self.data['SMA_buy'] = (self.data['LONG_SMA'] < self.data['SHORT_SMA']) & (self.data['prev_LONG_SMA'] > self.data['prev_SHORT_SMA'])
        self.data['SMA_sell'] = (self.data['LONG_SMA'] > self.data['SHORT_SMA']) & (self.data['prev_LONG_SMA'] < self.data['prev_SHORT_SMA'])
        self.data['MACD_buy'] = (self.data['MACD'] > self.data['Signal_Line']) & (self.data['prev_MACD'] < self.data['prev_Signal_Line'])
        self.data['MACD_sell'] = (self.data['MACD'] < self.data['Signal_Line']) & (self.data['prev_MACD'] > self.data['prev_Signal_Line'])
        self.data['SAR_buy'] = (self.data['SAR'] < self.data['Close']) & (self.data['prev_SAR'] > self.data['prev_Close'])
        self.data['SAR_sell'] = (self.data['SAR'] > self.data['Close']) & (self.data['prev_SAR'] < self.data['prev_Close'])
        self.data['ADX_buy'] = (self.data['+DI'] > self.data['-DI']) & (self.data['prev_+DI'] < self.data['prev_-DI']) & (self.data['ADX'] > 25)
        self.data['ADX_sell'] = (self.data['+DI'] < self.data['-DI']) & (self.data['prev_+DI'] > self.data['prev_-DI']) & (self.data['ADX'] > 25)
    
        for i, row in self.data.iterrows():
            buy_signals_count = sum(row[f'{indicator}_buy'] for indicator in self.active_indicators)
            sell_signals_count = sum(row[f'{indicator}_sell'] for indicator in self.active_indicators)
    
            total_active_indicators = len(self.active_indicators)
            if total_active_indicators <= 2:
                if buy_signals_count == total_active_indicators:
                    self._open_operation('long', row)
                elif sell_signals_count == total_active_indicators:
                    self._open_operation('short', row)
            else:
                if buy_signals_count > total_active_indicators / 2:
                    self._open_operation('long', row)
                elif sell_signals_count > total_active_indicators / 2:
                    self._open_operation('short', row)
    
            # Verifica y cierra operaciones basadas en stop_loss o take_profit
            self.check_close_operations(row)
    
        # Actualiza el valor de la estrategia en cada iteración
        total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
        self.strategy_value.append(total_value)
        
        return self.data, self.strategy_value

    
        
    def decide_operation(self, buy_signals_count, sell_signals_count, total_active_indicators):
        if total_active_indicators <= 2:
            if buy_signals_count == total_active_indicators:
                return 'long'
            elif sell_signals_count == total_active_indicators:
                return 'short'
        else:
            if buy_signals_count > total_active_indicators / 2:
                return 'long'
            elif sell_signals_count > total_active_indicators / 2:
                return 'short'
        return None

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

    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com)  # Decrementa el efectivo al cerrar la venta en corto, basado en el nuevo precio
                #print(f"Operación {op.operation_type} cerrada en {row.name}, Precio: {row['Close']}")    
                op.closed = True

    def calculate_operation_value(self, op, current_price):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0

    def plot_results(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()