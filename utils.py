import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
class TradingStrategy:
    def __init__(self, data_path):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.com = 0.00125
        self.strategy_value = [1_000_000]
        self.n_shares = 10
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal}
        }
        self.active_indicators = []

    def load_data(self, time_frame):
        file_mapping = {
            "5m": "data/aapl_5m_train.csv",
            "1h": "data/aapl_1h_train.csv",
            "1d": "data/aapl_1d_train.csv",
            "1m": "data/aapl_1m_train.csv"
        }
        file_name = file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70

    def sma_buy_signal(self, row, prev_row=None):
        return row.LONG_SMA < row.SHORT_SMA

    def sma_sell_signal(self, row, prev_row=None):
        return row.LONG_SMA > row.SHORT_SMA

    def macd_buy_signal(self, row, prev_row=None):
        
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
        return False  

    def macd_sell_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
        return False  

    def execute_trades(self):
        for i, row in self.data.iterrows():
            prev_row = self.data.iloc[i - 1] if i > 0 else None
            for indicator in self.active_indicators:
                if self.indicators[indicator]['buy'](row, prev_row):
                    self._open_operation('long', row)
                elif self.indicators[indicator]['sell'](row, prev_row) and self.operations:
                    self._close_operations(row, 'sell')

    def _open_operation(self, operation_type, row):
        self.operations.append(Operation(operation_type, row.Close, row.Timestamp, self.n_shares, row.Close * 0.95, row.Close * 1.05))
        self.cash -= row.Close * self.n_shares * (1 + self.com)

    def _close_operations(self, row, reason):
        if reason == 'sell':
            for op in self.operations:
                self.cash += row.Close * op.n_shares * (1 - self.com)
            self.operations.clear()
        
    def plot_results(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

