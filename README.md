
# Trading Strategy Project

## Description

The Trading Strategy Project is a comprehensive initiative aimed at developing, optimizing, and backtesting trading strategies using a variety of technical indicators across different timeframes. In the realm of financial markets, employing quantitative methods and algorithmic trading strategies has become increasingly essential for traders and investors seeking to gain a competitive edge and enhance their profitability.

This project entails a systematic approach to building effective trading strategies, beginning with the selection of technical indicators that offer insights into market trends, momentum, volatility, and other crucial aspects of price movements. These indicators serve as the foundation upon which buy and sell signals are defined, guiding the decision-making process for executing trades.

The project structure encompasses various stages of strategy development and evaluation. It includes organizing and preprocessing datasets for training and validation, implementing technical indicators, defining trading signals, generating combinations of indicators, and conducting rigorous backtesting to assess strategy performance. Through iterative optimization techniques, such as parameter tuning and strategy refinement, the goal is to enhance the effectiveness and robustness of the trading strategies.

The project leverages Python programming language and relevant libraries for data analysis, visualization, and strategy implementation. By adhering to best practices in project organization, documentation, and version control, the project ensures clarity, reproducibility, and collaboration among team members.

## Project Structure

- **data/**: Contains training and test datasets for different timeframes.
- **technical_analysis/**: Module-specific code to run the strategy.
- **utils/**: Helper methods and extra functions.
- **report.ipynb**: Jupyter notebook containing visualizations, tables, and conclusions.
- **venv/**: Virtual environment.
- **.gitignore**: Python's gitignore file from GitHub.
- **README.md**: Description of the project and instructions to run the main code.
- **requirements.txt**: Libraries and versions required to run the module.

## Usage

1. Set up a Python environment.
2. Install the required libraries listed in `requirements.txt`.
3. Run the main code in `technical_analysis/main.py`:

            3.1 **Loading Data**: The first step is to load the training data according to the specified time frame (5 minutes, 1 hour, 1 day, etc.). These data are contained in CSV files located in the `data/` directory. The `load_data` function handles this task by reading the corresponding file and storing the data in a Pandas DataFrame object.

            3.2 **Calculating Indicators**: Once the data is successfully loaded, relevant technical indicators are calculated. These indicators include RSI (Relative Strength Index), SMA (Simple Moving Average), MACD (Moving Average Convergence Divergence), SAR (Stop and Reverse), ADX (Average Directional Index), and Stochastic Oscillator. Each indicator is computed using historical price data and added as a new column to the data DataFrame.

            3.3 **Defining Buy/Sell Signals**: After calculating the indicators, buy and sell signals are defined for each indicator. These signals determine when to open or close a position in the market. For example, in the case of RSI, a buy signal may be generated when the RSI falls below a specific threshold (e.g., 30), indicating that the asset is oversold and may be a good time to buy. Conversely, a sell signal may be generated when the RSI rises above another threshold (e.g., 70), indicating that the asset is overbought and may be a good time to sell.

            3.4 **Generating Indicator Combinations**: With the buy and sell signals defined for each indicator, all possible combinations of indicators are generated. This involves creating subsets of indicators that will be used in the trading strategy. For example, if there are three indicators (RSI, SMA, and MACD), combinations will be generated that include only RSI, only SMA, only MACD, RSI and SMA, RSI and MACD, SMA and MACD, and RSI, SMA, and MACD.

            3.5 **Backtesting Strategies**: Once all indicator combinations have been generated, backtesting of each trading strategy is performed. This involves simulating the execution of buy and sell signals over the price history and evaluating the performance of the strategy in terms of profits and losses. During backtesting, open operations, cash balance, and portfolio value are tracked over time.

            3.6 **Optimizing Parameters**: In addition to backtesting, parameters of each indicator and the strategy as a whole are optimized. This involves adjusting parameter values (such as time periods, signal thresholds, etc.) to maximize the performance of the strategy. Techniques such as TPE (Tree-structured Parzen Estimator), Grid Search, PSO (Particle Swarm Optimization), or Genetic Algorithms are used to find the best parameter values.

            3.7 **Selecting Optimal Strategy**: After completing backtesting and parameter optimization, the optimal strategy is selected. This involves identifying the combination of indicators and parameter values that produce the best performance in terms of profits. The optimal strategy is chosen based on metrics such as return on investment, success rate, or any other criteria defined by the user.


4. Explore the results and conclusions in `report.ipynb`.


## Requirements

- Python 3.6 or higher
- Libraries listed in `requirements.txt`

## Authors
- Sanchez Soto Luis Eduardo
- Hernández Zatarain Sofía
- Robles Cobián Luis Ramón
- Castro González, Darío
- Zatarain Galindo Miguel Adolfo



## License

See the LICENSE.md file for details.

