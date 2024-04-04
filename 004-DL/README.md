
#  Machine Learning Trading Strategies

## Description

The Machine Learning Trading Strategies Project is an advanced initiative designed to harness the power of machine learning in financial markets. By applying Logistic Regression, Support Vector Machines (SVM), and XGBoost algorithms, the project aims to develop, optimize, and backtest robust trading strategies across various timeframes. In the fast-paced world of trading, these machine learning models offer a significant edge by analyzing market dynamics, recognizing patterns, and predicting future movements with greater accuracy.

The core of the project lies in its strategic approach to integrate technical indicators with machine learning models. Technical indicators provide key insights into market trends, momentum, and volatility, which are then used to craft predictive models for buy and sell signals. Logistic Regression is employed for its simplicity and effectiveness in binary classification problems, SVM is utilized for its capacity to handle non-linear data, and XGBoost is chosen for its speed and performance in dealing with structured data.

This comprehensive framework includes data preparation, feature engineering, model training, and rigorous backtesting to evaluate the performance of each trading strategy. Through continuous optimization and refinement, the project seeks to achieve the most profitable and reliable trading strategies.



## Project Structure

- **data/**: Directory containing the training and validation datasets for various timeframes for AAPL. 
- **machine_learning/**: Scripts for training and optimizing Logistic Regression, SVM, and XGBoost models.
- **utils/**: Utility functions to support data preprocessing, analysis, and other auxiliary tasks.
- **report.ipynb**: A Jupyter notebook with detailed visualizations and analysis of the trading strategies' performance.
- **venv/**: The project's virtual environment directory.
- **.gitignore**: Customized Python .gitignore file for the project.
- **README.md**: Comprehensive project documentation including setup and execution instructions.
- **requirements.txt**: A list of external libraries and dependencies required by the project.

## Usage

1. Set up a Python environment.
2. Install the required libraries listed in `requirements.txt`.
3. Run the main code in `machin_learning/main.py`:

    3.1 **Loading Data**: The process starts with loading the dataset corresponding to the desired timeframe (e.g., 5 minutes, 1 hour). Data are read from CSV files located in the `data/` directory, utilizing the `load_data` method to store the information in a Pandas DataFrame.

    3.2 **Calculating Indicators and Features**: After loading the data, the system calculates various technical indicators and statistical features such as returns, volatility, and price trends. These calculations form the basis for the subsequent machine learning models. The indicators include, but are not limited to, traditional ones like RSI and SMA, as well as derived features like price lags and volatility measures.

    3.3 **Defining Buy/Sell Signals**: Based on the calculated features, buy and sell signals are defined for each machine learning model (Logistic Regression, SVM, XGBoost). These signals are determined by specific conditions derived from the model predictions, indicating optimal points for entering or exiting trades.

    3.4 **Machine Learning Models**: The core of the strategy involves training Logistic Regression, SVM, and XGBoost models on the dataset. The models are optimized to predict buy and sell signals based on the financial data and indicators. This process involves hyperparameter tuning using Optuna to find the most effective model configurations.

    3.5 **Backtesting Strategies**: With the models trained and signals defined, the system backtests the trading strategies by simulating trades within the historical data. This process evaluates the performance of each strategy, tracking metrics such as portfolio value and cash balance over time.

    3.6 **Optimizing Trade Parameters**: Further optimization is conducted on trade parameters like stop-loss and take-profit thresholds. This step aims to fine-tune the strategy for maximum profitability, employing techniques like grid search or other optimization algorithms to identify the best parameter settings.

    3.7 **Selecting Optimal Strategy**: After thorough backtesting and parameter optimization, the best-performing strategy is selected based on criteria such as return on investment and overall profitability. This optimal strategy is then ready for real-world deployment or further testing on out-of-sample data.

4. **Visualizing Results**: The project includes functionality to plot the performance of the trading strategy, showcasing the growth of the portfolio value over time and illustrating the effectiveness of the selected approach.

5. **Testing with New Data**: The system provides capabilities to test the optimized strategy on new datasets, allowing for evaluation in different market conditions or timeframes.

6. Explore the results and conclusions in `report.ipynb`.


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

