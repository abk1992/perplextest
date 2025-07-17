import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Optional models
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="Advanced Stock Prediction", layout="wide")

@st.cache_data
def load_stock_list():
    with open("stocks.json", "r") as f:
        data = json.load(f)
    return data["stocks"]

def fix_ohlcv_columns(data):
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in data.columns.values]
    for base_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        candidates = [i for i in data.columns if base_col.lower() in i.lower()]
        if candidates and base_col not in data.columns:
            data[base_col] = data[candidates[0]]
    return data

def compute_enhanced_technical_indicators(df):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_Middle'] = bb_middle
    df['BB_Upper'] = bb_middle + (bb_std * 2)
    df['BB_Lower'] = bb_middle - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = np.where(bb_range != 0, (close - df['BB_Lower']) / bb_range, 0.5)
    low_min = low.rolling(window=14).min()
    high_max = high.rolling(window=14).max()
    stoch_range = high_max - low_min
    df['Stoch_K'] = np.where(stoch_range != 0, 100 * (close - low_min) / stoch_range, 50)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    df['Williams_R'] = np.where(stoch_range != 0, -100 * (high_max - close) / stoch_range, -50)
    volume_sma = volume.rolling(window=10).mean()
    df['Volume_SMA'] = volume_sma
    df['Volume_Ratio'] = np.where(volume_sma != 0, volume / volume_sma, 1)
    df['Price_Change'] = close.pct_change()
    df['Price_Change_5'] = close.pct_change(5)
    df['Volatility'] = df['Price_Change'].rolling(window=20).std()
    h_l = high - low
    h_c = np.abs(high - close.shift())
    l_c = np.abs(low - close.shift())
    tr = np.maximum(h_l, np.maximum(h_c, l_c))
    df['ATR'] = pd.Series(tr).rolling(window=14).mean()
    df['High_20'] = high.rolling(window=20).max()
    df['Low_20'] = low.rolling(window=20).min()
    typical_price = (high + low + close) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    tp_std = typical_price.rolling(window=20).std()
    df['CCI'] = np.where(tp_std != 0, (typical_price - tp_sma) / (0.015 * tp_std), 0)
    raw_money_flow = typical_price * volume
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    df['MFI'] = np.where(negative_flow != 0, 100 - (100 / (1 + positive_flow / negative_flow)), 100)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

class EnhancedLSTMModel:
    def __init__(self, n_input=30, n_output=10, n_features=20):
        self.n_input = n_input
        self.n_output = n_output
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
    def build_bidirectional_lstm(self):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.n_input, self.n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.n_output)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.n_input - self.n_output):
            X.append(data[i:i + self.n_input])
            y.append(data[i + self.n_input:i + self.n_input + self.n_output, 0])
        return np.array(X), np.array(y)
    def train(self, data):
        scaled_data = self.scaler.fit_transform(data)
        X, y = self.create_sequences(scaled_data)
        if len(X) == 0: return False, None, None
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        self.model = self.build_bidirectional_lstm()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[early_stopping], verbose=0)
        return history, X_test, y_test
    def predict(self, data):
        scaled_data = self.scaler.transform(data)
        last_seq = scaled_data[-self.n_input:]
        last_seq = np.expand_dims(last_seq, axis=0)
        forecast_scaled = self.model.predict(last_seq)[0]
        dummy = np.zeros((len(forecast_scaled), self.scaler.n_features_in_))
        dummy[:, 0] = forecast_scaled
        forecast = self.scaler.inverse_transform(dummy)[:, 0]
        return forecast

class EnsemblePredictor:
    def __init__(self, n_input=30, n_output=10):
        self.n_input = n_input
        self.n_output = n_output
        self.models = {}
        self.scalers = {}
    def prepare_data_for_ml(self, data):
        X, y = [], []
        for i in range(len(data) - self.n_input - self.n_output):
            X.append(data[i:i + self.n_input].flatten())
            y.append(data[i + self.n_input:i + self.n_input + self.n_output, 0])
        return np.array(X), np.array(y)
    def train_ensemble(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        self.scalers['main'] = scaler
        X, y = self.prepare_data_for_ml(scaled_data)
        if len(X) == 0: return False
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        models_to_train = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        if XGBRegressor is not None:
            models_to_train['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
        results = {}
        for name, model in models_to_train.items():
            try:
                model.fit(X_train, y_train[:, 0])
                self.models[name] = model
                pred = model.predict(X_test)
                mse = mean_squared_error(y_test[:, 0], pred)
                results[name] = {'mse': mse, 'model': model}
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
        return results
    def predict_ensemble(self, data, n_output):
        scaler = self.scalers['main']
        scaled_data = scaler.transform(data)
        last_seq = scaled_data[-self.n_input:].flatten().reshape(1, -1)
        predictions = {}
        for name, model in self.models.items():
            try:
                pred_scaled = model.predict(last_seq)[0]
                dummy = np.zeros((1, scaler.n_features_in_))
                dummy[0, 0] = pred_scaled
                pred = scaler.inverse_transform(dummy)[0, 0]
                predictions[name] = [pred] * n_output
            except Exception as e:
                st.warning(f"Prediction failed for {name}: {str(e)}")
        return predictions

def train_prophet_model(df):
    if Prophet is None:
        st.warning("Prophet not installed. Skipping Prophet model.")
        return None, None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close_col = [col for col in df.columns if 'Close' in str(col)][0]
        else:
            close_col = 'Close'
        prophet_df = df.reset_index()[['Date', close_col]].rename(columns={'Date': 'ds', close_col: 'y'})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        return model, forecast.tail(10)['yhat'].values
    except Exception as e:
        st.error(f"Prophet model failed: {str(e)}")
        return None, None

def train_arima_model(df):
    if ARIMA is None:
        st.warning("ARIMA not installed. Skipping ARIMA model.")
        return None, None
    try:
        close_col = 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            close_col = [col for col in df.columns if 'Close' in str(col)][0]
        close_prices = df[close_col].values
        model = ARIMA(close_prices, order=(5, 1, 0))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=10)
        return fitted_model, forecast
    except Exception as e:
        st.error(f"ARIMA model failed: {str(e)}")
        return None, None

def main():
    st.title('🚀 Robust Advanced Stock Price Prediction')
    st.markdown("---")
    with st.sidebar:
        st.header("📊 Configuration")
        stock_list = load_stock_list()
        ticker = st.selectbox('Stock Ticker', options=stock_list, index=0)
        start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))
        st.header("🔧 Model Settings")
        n_input = st.slider("Lookback Days", 20, 60, 30)
        n_output = st.slider("Forecast Days", 5, 15, 10)
        model_options = st.multiselect(
            "Select Models to Train",
            ['Enhanced LSTM', 'Ensemble ML', 'Prophet', 'ARIMA'],
            default=['Enhanced LSTM', 'Ensemble ML']
        )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header(f"📈 Analysis for {ticker}")
        with st.spinner('Loading and processing data...'):
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error("No data found for the given ticker and date range.")
                return
            data = fix_ohlcv_columns(data)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
                return
            data = compute_enhanced_technical_indicators(data)
            feature_columns = [
                'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
                'EMA_12', 'EMA_26', 'MACD', 'RSI', 'BB_Position',
                'Volume_Ratio', 'Price_Change', 'Volatility', 'ATR',
                'High_20', 'Low_20', 'CCI', 'MFI', 'Stoch_K', 'Williams_R'
            ]
            available_features = [col for col in feature_columns if col in data.columns]
            dataset = data[available_features].values
            st.success(f"✅ Data loaded successfully! Shape: {dataset.shape}")
            st.info(f"Available features: {len(available_features)}")
    with col2:
        st.header("📊 Quick Stats")
        if not data.empty and 'Close' in data.columns and 'Volume' in data.columns:
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].pct_change().iloc[-1] * 100
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            if 'RSI' in data.columns:
                st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
            if 'MACD' in data.columns:
                st.metric("MACD", f"{data['MACD'].iloc[-1]:.3f}")
    st.header("🤖 Model Training & Predictions")
    results = {}
    if st.button("🚀 Train Models & Generate Predictions"):
        if len(available_features) < 5:
            st.error("Not enough features available for training. Please check the data.")
            return
        with st.spinner('Training models...'):
            if 'Enhanced LSTM' in model_options:
                try:
                    lstm_model = EnhancedLSTMModel(n_input=n_input, n_output=n_output, n_features=len(available_features))
                    history, X_test, y_test = lstm_model.train(dataset)
                    if history:
                        lstm_predictions = lstm_model.predict(dataset)
                        results['Enhanced LSTM'] = lstm_predictions
                        y_pred = lstm_model.model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
                        mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
                        st.success(f"✅ Enhanced LSTM trained! RMSE: {rmse:.3f}, MAE: {mae:.3f}")
                except Exception as e:
                    st.error(f"Enhanced LSTM failed: {str(e)}")
            if 'Ensemble ML' in model_options:
                try:
                    ensemble = EnsemblePredictor(n_input=n_input, n_output=n_output)
                    ensemble_results = ensemble.train_ensemble(dataset)
                    if ensemble_results:
                        ensemble_predictions = ensemble.predict_ensemble(dataset, n_output=n_output)
                        if ensemble_predictions:
                            for name, pred_list in ensemble_predictions.items():
                                results[f'Ensemble: {name}'] = pred_list
                            avg_pred = np.mean([pred_list[0] for pred_list in ensemble_predictions.values()])
                            results['Ensemble ML (avg)'] = [avg_pred] * n_output
                            st.success(f"✅ Ensemble models trained! Models: {list(ensemble_predictions.keys())}")
                except Exception as e:
                    st.error(f"Ensemble ML failed: {str(e)}")
            if 'Prophet' in model_options:
                prophet_model, prophet_pred = train_prophet_model(data)
                if prophet_pred is not None:
                    results['Prophet'] = prophet_pred
                    st.success("✅ Prophet model trained!")
            if 'ARIMA' in model_options:
                arima_model, arima_pred = train_arima_model(data)
                if arima_pred is not None:
                    results['ARIMA'] = arima_pred
                    st.success("✅ ARIMA model trained!")
    if results:
        st.header("📊 Prediction Results")
        predictions_df = pd.DataFrame()
        for model_name, preds in results.items():
            if isinstance(preds, (list, np.ndarray)):
                predictions_df[model_name] = preds[:n_output]
        if not predictions_df.empty:
            predictions_df.index = range(1, len(predictions_df) + 1)
            predictions_df.index.name = 'Day'
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_output, freq="B")
            predictions_df.insert(0, "Prediction Date", forecast_dates)
            # --- Add % change columns ---
            current_price = data['Close'].iloc[-1]
            for col in predictions_df.columns:
                if col != "Prediction Date":
                    predictions_df[f"{col} %Change"] = 100 * (predictions_df[col] - current_price) / current_price
            st.dataframe(predictions_df.round(2))
            fig, ax = plt.subplots(figsize=(12, 6))
            recent = np.array(data['Close'].values[-n_input:]).flatten()
            forecast_avg = np.mean(
                [preds[:n_output] for mn, preds in results.items() if "LSTM" in mn or "avg" in mn or "Prophet" in mn or "ARIMA" in mn],
                axis=0
            ) if any(mn for mn in results if "LSTM" in mn or "avg" in mn or "Prophet" in mn or "ARIMA" in mn) else np.zeros(n_output)
            all_days = np.concatenate([recent, forecast_avg])
            ax.plot(range(len(all_days)), all_days, label='Price + Forecast', linewidth=2)
            ax.axvline(x=n_input - 1, color='red', linestyle='--', alpha=0.7, label='Forecast Start')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{ticker} Stock Price Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            csv = predictions_df.to_csv()
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv"
            )
    st.header("ℹ️ Technical Indicators Included")
    with st.expander("📚 Available Technical Indicators"):
        st.markdown("""
        **Price Indicators**: SMA (5,10,20), EMA (12,26), MACD, Bollinger Bands  
        **Momentum**: RSI, Stochastic, Williams %R, CCI  
        **Volume**: Volume SMA/Ratio, MFI  
        **Volatility**: ATR, Price Volatility  
        **Support/Resistance**: 20-day High/Low
        """)
    with st.expander("⚠️ Important Disclaimers"):
        st.warning("""
        - This tool is for educational purposes only and should not be used for actual trading decisions.
        - Stock market predictions are inherently uncertain and past performance doesn't guarantee future results.
        - Always consult with financial professionals before making investment decisions.
        - The models may not perform well during high volatility or unusual market conditions.
        """)

if __name__ == "__main__":
    main()
