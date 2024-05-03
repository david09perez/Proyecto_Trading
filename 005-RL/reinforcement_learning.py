import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from technical_analysis import Operation, TradingStrategy
import ta


def calculate_indicators(self):
    # Lista de ventanas de tiempo para el cálculo de cada indicador
    rsi_windows = [5, 10, 14, 20, 25]
    sma_windows = [(5, 21), (10, 30), (20, 50)]
    macd_settings = [(12, 26, 9), (5, 35, 5)]
    sar_settings = [(0.02, 0.2), (0.01, 0.1)]
    adx_window = [14, 20, 28]
    stoch_window = [(14, 3), (10, 3), (20, 4)]

    # Calcular RSI para diferentes ventanas
    for window in rsi_windows:
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window)
        self.data[f'RSI_{window}'] = rsi_indicator.rsi()

    # Calcular SMA para diferentes combinaciones de ventanas cortas y largas
    for short_window, long_window in sma_windows:
        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=short_window)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=long_window)
        self.data[f'SHORT_SMA_{short_window}'] = short_ma.sma_indicator()
        self.data[f'LONG_SMA_{long_window}'] = long_ma.sma_indicator()

    # Calcular MACD con diferentes configuraciones
    for fast, slow, sign in macd_settings:
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=sign)
        self.data[f'MACD_{fast}_{slow}_{sign}'] = macd.macd()
        self.data[f'MACD_signal_{fast}_{slow}_{sign}'] = macd.macd_signal()

    # Calcular Parabolic SAR para diferentes configuraciones
    for step, max_step in sar_settings:
        sar = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], step=step, max_step=max_step)
        self.data[f'SAR_{step}_{max_step}'] = sar.psar()

    # Calcular ADX para diferentes ventanas
    for window in adx_window:
        adx_indicator = ta.trend.ADXIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=window)
        self.data[f'ADX_{window}'] = adx_indicator.adx()
        self.data[f'+DI_{window}'] = adx_indicator.adx_pos()
        self.data[f'-DI_{window}'] = adx_indicator.adx_neg()

    # Calcular Stochastic Oscillator para diferentes configuraciones
    for k_window, d_window in stoch_window:
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=k_window, smooth_window=d_window)
        self.data[f'stoch_%K_{k_window}_{d_window}'] = stoch_indicator.stoch()
        self.data[f'stoch_%D_{k_window}_{d_window}'] = stoch_indicator.stoch_signal()

    # Eliminar valores NA y reiniciar el índice
    self.data.dropna(inplace=True)
    self.data.reset_index(drop=True, inplace=True)



class TradingEnv(gym.Env):
    """Un ambiente de trading personalizado para Gymnasium."""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnv, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = 0

        # Acciones del espacio: [0 = vender, 1 = mantener, 2 = comprar]
        self.action_space = spaces.Discrete(3)

        # Ejemplo de espacio de estado: saldo, cantidad de acciones poseídas, precio actual y algunos indicadores técnicos
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.shares_bought = 0
        self.shares_sold = 0
        self.current_step = 0

        return self._next_observation()

    def step(self, action):
        # Ejecuta una acción
        self._take_action(action)
        self.current_step += 1

        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward()

        obs = self._next_observation()

        return obs, reward, done, {}

    def _next_observation(self):
        # Obtiene datos del siguiente paso
        frame = np.append(self.data.iloc[self.current_step].values, [
            self.balance,
            self.shares_held
        ])

        return frame

    def _take_action(self, action):
        # Define cómo se deben manejar las acciones
        current_price = self.data.iloc[self.current_step]['Close']

        if action == 0:  # Vender
            self._sell(current_price)
        elif action == 2:  # Comprar
            self._buy(current_price)

    def _buy(self, price):
        # Define lógica para comprar
        if self.balance > price:
            self.shares_bought = self.balance / price
            self.shares_held += self.shares_bought
            self.balance -= self.shares_bought * price
            self.balance -= self.transaction_cost * (self.shares_bought * price)

    def _sell(self, price):
        # Define lógica para vender
        if self.shares_held > 0:
            self.shares_sold = self.shares_held
            self.balance += self.shares_sold * price
            self.shares_held = 0
            self.balance -= self.transaction_cost * (self.shares_sold * price)

    def _calculate_reward(self):
        # Calcula la recompensa
        return self.net_worth - self.initial_balance  # simple diferencia de valor neto

    def render(self, mode='human', close=False):
        # Renderiza el ambiente si es necesario
        pass
















