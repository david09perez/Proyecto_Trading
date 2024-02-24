import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

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
        self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
class TradingStrategy:
    def __init__(self, data_path):
        
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
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal}
        }
        self.active_indicators = []

    def load_data(self, time_frame):
        
        """
        Loads historical stock data from the CSV file corresponding to the specified time frame. This function
        ensures that the data is available and in the correct format for analysis. It raises an error if the
        time frame is unsupported, ensuring that the strategy operates on valid data sets.
        
        Parameters:
            time_frame: A string identifier for the desired time frame of the stock data.
            
        """
        
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
        
        """
        Activates a specified technical indicator for use in the trading strategy by adding it to the list of active indicators.
        This allows the strategy to dynamically include or exclude indicators based on their performance or relevance to current
        market conditions. 

        Parameters:
            indicator_name: The name of the indicator to be activated. This name must match one of the keys in the
                            strategy's `indicators` dictionary, which contains all available indicators and their
                            corresponding buy/sell signal functions.
                                  
        """
        
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

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
        
        return row.LONG_SMA < row.SHORT_SMA

    def sma_sell_signal(self, row, prev_row=None):
        
        """
        Generates a sell signal based on SMA crossovers. A sell signal occurs when the short-term SMA crosses below the long-term
        SMA, suggesting a downward price momentum and a potential bearish trend. This function relies on both the current and 
        previous rows to detect the crossover event.
        
        Parameters:
            row: The current row of market data.
            prev_row: The previous row of market data.
            
        """
        
        return row.LONG_SMA > row.SHORT_SMA

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
    
    #def sar_buy_signal(self, row, prev_row=None):
        
        #"""
        #Generates a buy signal based on the Parabolic SAR indicator. A buy signal is generated when the SAR value is
        #below the current price, indicating a potential uptrend. The function checks the SAR values against the close
        #prices of the current and previous rows to determine the signal.
        
        #Parameters:
            #row: The current row of market data.
            #prev_row: The previous row of market data.
            
        #"""
        
        #return prev_row is not None and row['SAR'] < row['Close'] and prev_row['SAR'] > prev_row['Close']

    #def sar_sell_signal(self, row, prev_row=None):
        
        #"""
        #Generates a sell signal based on the Parabolic SAR indicator. A sell signal is generated when the SAR value is
        #above the current price, indicating a potential downtrend. The function evaluates the SAR values against the
        #close prices of the current and previous rows to generate the signal.
        
        #Parameters:
            #row: The current row of market data.
            #prev_row: The previous row of market data.
            
        #"""
        
        #return prev_row is not None and row['SAR'] > row['Close'] and prev_row['SAR'] < prev_row['Close']

    #def adx_buy_signal(self, row, prev_row=None):
        
        #"""
        #Generates a buy signal when the ADX indicates a strong upward trend, which is when the +DI line crosses above
        #the -DI line and the ADX value is above a certain threshold, suggesting a strengthening trend. The
        #function requires both the current and previous rows to identify the crossover and confirm the trend strength.
        
        #Parameters:
            #row: The current row of market data.
            #prev_row: The previous row of market data.
            
        #"""
        
        #return prev_row is not None and row['+DI'] > row['-DI'] and row['ADX'] > 25 and prev_row['+DI'] < prev_row['-DI']

    #def adx_sell_signal(self, row, prev_row=None):
        
        #"""
        #Generates a sell signal when the ADX indicates a strong downward trend, which is when the +DI line crosses below
        #the -DI line and the ADX value is above a certain threshold, indicating a strong bearish trend. The
        #function uses data from both the current and previous rows to detect the crossover and assess the trend's strength.
        
        #Parameters:
            #row: The current row of market data.
            #prev_row: The previous row of market data.
            
        #"""
        
        #return prev_row is not None and row['+DI'] < row['-DI'] and row['ADX'] > 25 and prev_row['+DI'] > prev_row['-DI'] 

    #def execute_trades(self):
        
         #"""
        #Iterates through each row of the dataset and evaluates the buy and sell conditions for each active indicator.
        #If a buy signal is detected based on the conditions defined for an indicator, a new long operation is initiated.
        
        #Conversely, if a sell signal is detected and there are open operations, the strategy closes these
        #operations based on the sell conditions. This function is central to executing the trading strategy, as it
        #directly translates the signals generated by the technical indicators into actionable trades.

        #"""
  
        #for i, row in self.data.iterrows():
            #prev_row = self.data.iloc[i - 1] if i > 0 else None
            #for indicator in self.active_indicators:
                #if self.indicators[indicator]['buy'](row, prev_row):
                    #self._open_operation('long', row)
                #elif self.indicators[indicator]['sell'](row, prev_row) and self.operations:
                    #self._close_operations(row, 'sell')

    def _open_operation(self, operation_type, row):
        
        """
        Opens a new trading operation based on the given operation type, long or short, and the market data at
        the current row. It sets the stop loss and take profit levels based on the current price and updates the cash
        balance to reflect the cost of the trade. The new operation is then added to the list of ongoing operations.
        
        Parameters:
            operation_type: The type of operation to open, long or short.
            row: The current row of market data used to determine the trade's entry price.
            
        """
        
        self.operations.append(Operation(operation_type, row.Close, row.Timestamp, self.n_shares, row.Close * 0.95, row.Close * 1.05))
        self.cash -= row.Close * self.n_shares * (1 + self.com)

    def _close_operations(self, row, reason):
        
        """
        Closes all open trading operations based on a specified reason. This method iterates through the list of open 
        operations and simulates selling the shares at the current row's closing price, thereby increasing the cash
        balance by the value of the sold shares minus the transaction costs. 
        
        After closing the operations, it clears the list of operations to reflect that there are no longer any open positions.

        Parameters:
            row: The row of market data from which to extract the closing price for selling the shares.
            reason: The reason for closing the operations.

        """
        
        if reason == 'sell':
            for op in self.operations:
                self.cash += row.Close * op.n_shares * (1 - self.com)
            self.operations.clear()
        
    def plot_results(self):
        
        """
        Visualizes the performance of the trading strategy by plotting the value of the strategy over time. This function
        generates a line chart reflecting the cumulative effect of the trades executed by the strategy.

        """
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

