import pandas as pd
import ta
      
class TradingEnv:
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
        self.indicators = {}
        self.calculate_indicators()
        
        self.current_step = 0
        self.last_price = 0
        self.last_action = None
        self.waiting_steps = 0
        
    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def calculate_indicators(self):
        # Lista de ventanas de tiempo para el cálculo de cada indicador
        rsi_windows = [5, 10, 14, 20, 25]
        sma_windows = [(5, 21), (10, 30), (20, 50)]
        macd_settings = [(12, 26, 9), (5, 35, 5)]
        sar_settings = [(0.02, 0.2), (0.01, 0.1)]
        adx_window = [14, 20, 28]
        stoch_window = [(14, 3), (10, 3), (20, 4)]

        for window in rsi_windows:
            rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window)
            self.data[f'RSI_{window}'] = rsi_indicator.rsi()

        for short_window, long_window in sma_windows:
            short_ma = ta.trend.SMAIndicator(self.data['Close'], window=short_window)
            long_ma = ta.trend.SMAIndicator(self.data['Close'], window=long_window)
            self.data[f'SHORT_SMA_{short_window}'] = short_ma.sma_indicator()
            self.data[f'LONG_SMA_{long_window}'] = long_ma.sma_indicator()

        for fast, slow, sign in macd_settings:
            macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=sign)
            self.data[f'MACD_{fast}_{slow}_{sign}'] = macd.macd()
            self.data[f'MACD_signal_{fast}_{slow}_{sign}'] = macd.macd_signal()

        for step, max_step in sar_settings:
            sar = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], step=step, max_step=max_step)
            self.data[f'SAR_{step}_{max_step}'] = sar.psar()

        for window in adx_window:
            adx_indicator = ta.trend.ADXIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=window)
            self.data[f'ADX_{window}'] = adx_indicator.adx()
            self.data[f'+DI_{window}'] = adx_indicator.adx_pos()
            self.data[f'-DI_{window}'] = adx_indicator.adx_neg()

        for k_window, d_window in stoch_window:
            stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=k_window, smooth_window=d_window)
            self.data[f'stoch_%K_{k_window}_{d_window}'] = stoch_indicator.stoch()
            self.data[f'stoch_%D_{k_window}_{d_window}'] = stoch_indicator.stoch_signal()

        self.data.dropna(inplace=True)
        drop_cols = list(self.data.columns)[:4]
        self.data = self.data.drop(columns = drop_cols)
        self.data.reset_index(drop=True, inplace=True)


    def reset(self):
        self.current_step = 0
        self.last_price = self.data['Close'][self.current_step]
        self.last_action = None
        return self.data.iloc[self.current_step].values    
    
    def step(self, action):
        current_price = self.data['Close'][self.current_step]
        price_change = current_price - self.last_price
        reward = 0

        if self.last_action is not None:
            if action == self.last_action:  # Mantener la misma acción
                if self.last_action == 2:  # Long
                    reward = price_change  # Recompensa o penalización directa basada en el cambio de precio
                elif self.last_action == 0:  # Short
                    reward = -price_change  # Recompensa o penalización directa basada en el cambio de precio inverso
                elif self.last_action == 1:
                    reward -= (self.data.Close.diff().abs().mean() * self.waiting_steps) * 0.1  # Penalización por inactividad si el mercado se mueve
                    self.waiting_steps += 1
            else:
                reward -= self.data.Close.diff().abs().mean()   # Penalización por cambiar de acción

        self.last_price = current_price
        self.last_action = action

        # Verificación de término de episodio
        self.current_step += 1
        done = self.current_step >= len(self.data)

        return self.data.iloc[self.current_step % len(self.data)].values, reward, done, {}


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        