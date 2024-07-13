import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from optuna.integration import TFKerasPruningCallback
import ta, joblib, os, optuna

"""Every week it is recommended that the parameters be retuned (delete the best_params.pkl and run the program again)"""

# Function to load and prepare data
def load_and_prepare_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Add Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(data['Close'])
    data['BB_high'] = bb_indicator.bollinger_hband()
    data['BB_low'] = bb_indicator.bollinger_lband()
    data['BB_mid'] = bb_indicator.bollinger_mavg()
    data['BB_width'] = (data['BB_high'] - data['BB_low']) / data['BB_mid']
    
    # Add VIX
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    data['VIX'] = vix_data['Close']
    
    # Additional indicators
    data['Returns'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    
    data.dropna(inplace=True)
    return data

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 3])  # Close price is the 4th column (index 3)
    return np.array(X), np.array(y)

# Load data
ticker = '^GSPC'  # S&P 500
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years of data
data = load_and_prepare_data(ticker, start_date, end_date)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
seq_length = 20
X, y = create_sequences(scaled_data, seq_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=False)

# Load the best hyperparameters if they exist, otherwise optimize and save them
best_params_path = 'best_params.pkl'

if os.path.exists(best_params_path):
    best_params = joblib.load(best_params_path)
else:
    # Define objective function for Optuna
    def objective(trial):
        model = Sequential()
        model.add(Bidirectional(LSTM(
            units=trial.suggest_int('units_1', 32, 128, step=32),
            return_sequences=True, 
            input_shape=(X_train.shape[1], X_train.shape[2])
        )))
        model.add(Dropout(trial.suggest_float('dropout_1', 0.2, 0.5, step=0.1)))
        model.add(Bidirectional(LSTM(
            units=trial.suggest_int('units_2', 32, 128, step=32), 
            return_sequences=False
        )))
        model.add(Dropout(trial.suggest_float('dropout_2', 0.2, 0.5, step=0.1)))
        model.add(Dense(trial.suggest_int('dense_units', 16, 64, step=16)))
        model.add(Dense(1))
        
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, 
                  validation_split=0.1, 
                  epochs=100, 
                  batch_size=32, 
                  callbacks=[early_stopping, TFKerasPruningCallback(trial, 'val_loss')], 
                  verbose=0)
        
        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss

    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters and save them
    best_params = study.best_params
    joblib.dump(best_params, best_params_path)

# Build and train the best model
best_model = Sequential()
best_model.add(Bidirectional(LSTM(
    units=best_params['units_1'], 
    return_sequences=True, 
    input_shape=(X_train.shape[1], X_train.shape[2])
)))
best_model.add(Dropout(best_params['dropout_1']))
best_model.add(Bidirectional(LSTM(
    units=best_params['units_2'], 
    return_sequences=False
)))
best_model.add(Dropout(best_params['dropout_2']))
best_model.add(Dense(best_params['dense_units']))
best_model.add(Dense(1))

best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Inverse transform the predictions and actual values
dummy_array = np.zeros((len(y_pred), scaled_data.shape[1]))
dummy_array_pred = dummy_array.copy()
dummy_array_pred[:, 3] = y_pred.flatten()  # Only set the close price predictions
dummy_array_actual = dummy_array.copy()
dummy_array_actual[:, 3] = y_test.flatten()  # Only set the actual close prices

# Perform the inverse transform
y_pred_inv = scaler.inverse_transform(dummy_array_pred)[:, 3]
y_test_inv = scaler.inverse_transform(dummy_array_actual)[:, 3]

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

# Calculate Directional Accuracy
direction_pred = np.sign(y_pred_inv[1:] - y_pred_inv[:-1])
direction_true = np.sign(y_test_inv[1:] - y_test_inv[:-1])
directional_accuracy = np.mean(direction_pred == direction_true) * 100

print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Plot predictions vs actual values
plt.figure(figsize=(16, 8))
plt.plot(data.index[-len(y_test):], y_test_inv, label='Actual Test Data', color='blue')
plt.plot(data.index[-len(y_pred):], y_pred_inv, label='Predicted Test Data', color='green')
plt.title(f'{ticker} LSTM Prediction - Test Set Comparison')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Forecast future prices
last_sequence = scaled_data[-seq_length:]
forecast = []
num_preds = 20 # Change this to the number of predictions you want
for _ in range(num_preds):
    next_pred = best_model.predict(last_sequence.reshape(1, seq_length, -1))
    forecast.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1, axis=0)
    last_sequence[-1, 3] = next_pred[0, 0]

# Inverse transform forecasted values
forecast_full = np.zeros((len(forecast), scaled_data.shape[1]))
forecast_full[:, 3] = forecast
forecast_full = scaler.inverse_transform(forecast_full)
forecast = forecast_full[:, 3]

# Plot actual and forecasted prices
plt.figure(figsize=(16, 8))
plt.plot(data.index[-50:], data['Close'][-50:], label='Actual Data', color='blue')
future_dates = pd.date_range(start=data.index[-1], periods=num_preds + 1, inclusive='right')
plt.plot(future_dates, forecast, label='Forecasted Data', color='red')
plt.title(f'{ticker} LSTM Prediction - Future Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
