import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def calcular_rendimiento_pasiva(csv_file='aapl_1d_train.csv', initial_cash=1000000):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    data = data.dropna()
    
    # Convert the 'Date' column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])

    # Get the closing price of the first and last data
    first_close = data.iloc[0]['Close']
    last_close = data.iloc[-1]['Close']

    # Calculate the asset's return
    passive_return = (last_close - first_close) / first_close

    print("The asset's return from the first closing to the last closing is: {:.2%}".format(passive_return))
    
    # Comparison with the strategy used
    cashfinal = 959594.3958  # This value was hardcoded; you should provide the logic or the correct value
    strategy_return = (cashfinal - initial_cash) / initial_cash
    
    print("The strategy's return from the first closing to the last closing is: {:.2%}".format(strategy_return))
    
    # Sort the data by date if not already sorted
    data = data.sort_values(by='Date')
    plt.figure(figsize=(12, 8))
    plt.plot(data['Date'], data['Close'], label='Closing Price', color='blue')
    plt.title('Closing Price of the Asset')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate the cumulative return
    data['Returns'] = data['Close'].pct_change().fillna(0)

    # Calculate the cumulative value
    data['Investment_Value'] = (1 + data['Returns']).cumprod() * initial_cash
    plt.figure(figsize=(12, 8))
    plt.plot(data['Date'], data['Investment_Value'], label='Investment Value', color='green')
    plt.title('Investment Performance')
    plt.xlabel('Date')
    plt.ylabel('Investment Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    final_value = data['Investment_Value'].iloc[-1]
    print("The final value of the investment: ${:,.2f}".format(final_value))
    
    # Passive Strategy vs Machine Learning Strategy
    print("The asset's return from the first closing to the last closing is: {:.2%}".format(passive_return))
    print("The strategy's return from the first closing to the last closing is: {:.2%}".format(strategy_return))
    passive_vs_strategy = passive_return - strategy_return
    print("Difference between the passive strategy and the machine learning strategy is: {:.2%}".format(passive_vs_strategy))


