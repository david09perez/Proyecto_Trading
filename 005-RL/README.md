

# Deep Learning Trading Strategies

## Description

The Deep Learning Trading Strategies Project leverages advanced deep learning models to predict and execute trading decisions in the financial markets. This project utilizes Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and Dense Neural Networks (DNNs) to analyze time-series data, identify patterns, and make informed predictions about future market movements. By integrating these models with trading indicators, the project aims to develop robust, automated trading strategies that can outperform traditional methods.

## Project Structure

- **data/**: Contains the training and validation datasets.
- **deep_learning/**: Houses the scripts for initializing and training the deep learning models.
- **utils/**: Includes utility functions for data preprocessing, feature calculation, and other support tasks.
- **report.ipynb**: A Jupyter notebook presenting detailed visualizations and analysis of the trading strategies' performance.
- **venv/**: Directory for the project's Python virtual environment.
- **.gitignore**: Custom Python .gitignore file.
- **README.md**: Documentation for setting up and running the project.
- **requirements.txt**: Lists external libraries and dependencies.

## Usage

1. **Environment Setup**:
   - Ensure Python 3.6 or higher is installed.
   - Set up a virtual environment and activate it.
   - Install required dependencies using `pip install -r requirements.txt`.

2. **Data Preparation**:
   - Load and preprocess data from the `data/` directory.
   - Calculate technical indicators and features needed for model training.

3. **Model Training and Evaluation**:
   - Train CNN, LSTM, and DNN models on the training data.
   - Evaluate models using validation data to adjust parameters and improve accuracy.

4. **Trade Execution**:
   - Use trained models to predict buy and sell signals based on real-time or historical data.
   - Execute trades automatically based on these predictions, adjusting for factors like stop loss and take profit.

5. **Performance Analysis**:
   - Backtest trading strategies to assess their potential profitability.
   - Analyze and visualize the performance of strategies using `report.ipynb`.

6. **Optimization**:
   - Use Optuna or similar frameworks to fine-tune model parameters and trading strategies for optimal performance.

## Key Features

- **Deep Learning Models**: Utilizes CNNs, LSTMs, and DNNs to predict market movements.
- **Automated Trading**: Automatically executes trades based on model predictions.
- **Backtesting**: Simulates trading strategies on historical data to evaluate their effectiveness.
- **Performance Optimization**: Applies hyperparameter tuning to enhance model and strategy performance.

## Requirements

- Python 3.6+
- See `requirements.txt` for library dependencies.

## Running the Project

Execute the main script after setting up the environment and installing dependencies:

```bash
python deep_learning/main.py
```

This will initiate the data loading, model training, and trading simulations based on configured parameters.

## Authors

- Sanchez Soto Luis Eduardo
- Hernández Zatarain Sofía
- Robles Cobián Luis Ramón
- Castro González, Darío
- Zatarain Galindo Miguel Adolfo

## License

Refer to the LICENSE.md file for licensing details.

This README provides clear instructions and a comprehensive overview tailored to your project's focus on deep learning, ensuring anyone reviewing or working with your project has a structured guide to follow.