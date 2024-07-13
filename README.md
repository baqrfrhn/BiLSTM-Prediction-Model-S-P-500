# S&P 500 LSTM Prediction Model

This repository contains Python code for predicting the daily closing prices of the S&P 500 index using LSTM (Long Short-Term Memory) neural networks. The model is built using TensorFlow/Keras and optimized using Optuna for hyperparameter tuning.

## Features

- **Data Loading and Preparation**: Historical data from Yahoo Finance is loaded and preprocessed. Additional technical indicators such as Bollinger Bands, VIX, Moving Averages, RSI, and MACD are computed and integrated into the dataset.
  
- **Model Architecture**: Bidirectional LSTM layers are utilized for capturing both past and future context in the sequences. Dropout layers are added for regularization to prevent overfitting. The model's hyperparameters (e.g., number of units, dropout rates, learning rate) are optimized using Optuna.

- **Evaluation Metrics**: The model performance is evaluated using metrics such as Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Directional Accuracy on a test set.

- **Forecasting**: Future prices are forecasted using the trained model, and the results are plotted alongside actual historical data to visualize the predictions.

## Usage

1. **Dependencies**: Ensure you have Python 3.x installed with the necessary libraries listed in `requirements.txt`. You can install them using `pip install -r requirements.txt`.

2. **Running the Code**: Modify the parameters in `main.py` (e.g., `ticker`, `seq_length`, `num_preds`) as per your requirements. Run the script using `python main.py`.

3. **Output**: After running, the script will output performance metrics and plots comparing actual vs. predicted prices. Future price forecasts will also be plotted.

## File Structure

- `main.py`: Main script that loads data, defines the LSTM model, trains and evaluates the model, and generates predictions.
  
- `utils.py`: Utility functions for data loading, preprocessing, sequence creation, and inverse transformations.

- `best_params.pkl`: Pickled file containing the best hyperparameters obtained from Optuna optimization.

- `readme.md`: This file providing an overview of the project, usage instructions, and explanation of files.

## Requirements

- Python 3.x
- numpy
- pandas
- yfinance
- matplotlib
- scikit-learn
- tensorflow
- optuna
- ta (Technical Analysis Library)

## Acknowledgments

The code is inspired by various tutorials and examples on using LSTM for time series forecasting, particularly with financial data.

## License

This project is licensed under the MIT License.
