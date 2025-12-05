import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class IndianStockPredictor:
    def __init__(self):
        # keys will be normalized symbols (e.g., RELIANCE.NS)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def normalize_symbol(self, symbol):
        """Return normalized symbol (append .NS by default if missing)"""
        symbol = symbol.strip().upper()
        if not symbol.endswith(('.NS', '.BO')):
            symbol = symbol + '.NS'
        return symbol

    def get_stock_data(self, symbol, period="2y"):
        """
        Fetch stock data; returns DataFrame or None.
        Use normalized symbol internally (e.g., RELIANCE.NS).
        """
        try:
            symbol_full = self.normalize_symbol(symbol)
            df = yf.download(symbol_full, period=period, progress=False)
            if df is None or df.empty:
                raise ValueError(f"No data found for symbol: {symbol_full}")
            # Ensure columns we expect exist
            df = df.rename(columns=str.capitalize)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators in-place and return df"""
        df = df.copy()
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        return df

    def create_features(self, df, forecast_days=30, create_target=True):
        """
        Create features. If create_target is True, adds 'Target' (Close shifted -forecast_days)
        If create_target is False (for prediction), it will not add Target nor shift; just builds features.
        Returns (df_processed, feature_columns)
        """
        df = df.copy()
        # must already have technical indicators
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'HL_PCT', 'PCT_change', 'MA5', 'MA20', 'MA50',
            'MACD', 'RSI', 'BB_Width', 'Volume_Ratio', 'ROC'
        ]

        # lag features
        for i in range(1, 6):
            df[f'Close_lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
            feature_columns.extend([f'Close_lag_{i}', f'Volume_lag_{i}'])

        if create_target:
            df['Target'] = df['Close'].shift(-forecast_days)
            df = df.dropna()
        else:
            # when predicting, we want the latest row(s). still need to drop rows with NaN in features,
            # but avoid dropping the last rows due to target shift.
            df = df.dropna(subset=feature_columns)

        return df, feature_columns

    def train_models(self, symbol, forecast_days=30, period="2y"):
        """Train models for a symbol. Stores models under normalized symbol key."""
        symbol_norm = self.normalize_symbol(symbol)

        stock_data = self.get_stock_data(symbol_norm, period=period)
        if stock_data is None:
            return False

        stock_data = self.calculate_technical_indicators(stock_data)
        stock_data, feature_columns = self.create_features(stock_data, forecast_days, create_target=True)

        if len(stock_data) < 200:
            print(f"Warning: only {len(stock_data)} rows available after feature creation. Model may be unreliable.")

        X = stock_data[feature_columns]
        y = stock_data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Decide which models need scaling
        models_to_scale = ['Linear Regression', 'SVR']

        scaler = StandardScaler()
        scaler.fit(X_train)  # fit on train only
        self.scalers[symbol_norm] = scaler

        # initialize models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }

        results = {}

        for name, model in models.items():
            if name in models_to_scale:
                X_train_model = scaler.transform(X_train)
                X_test_model = scaler.transform(X_test)
            else:
                X_train_model = X_train.values
                X_test_model = X_test.values

            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            # keep the same 'accuracy' if you like, but R2 is more standard
            accuracy = max(0, 100 * (1 - mae / (y_test.mean() + 1e-9)))

            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Accuracy(est): {accuracy:.2f}%")

        self.models[symbol_norm] = {'results': results, 'feature_columns': feature_columns, 'forecast_days': forecast_days}

        # feature importance for RF
        if 'Random Forest' in results:
            rf = results['Random Forest']['model']
            feature_imp = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance[symbol_norm] = feature_imp
            print("\nTop 10 Important Features:")
            print(feature_imp.head(10))

        return True

    def predict_future_price(self, symbol, days_ahead=30, period="2y"):
        """Predict future price using trained models. Returns dict of model_name -> predicted price"""
        symbol_norm = self.normalize_symbol(symbol)

        if symbol_norm not in self.models:
            print(f"Model for {symbol_norm} not trained. Training now...")
            if not self.train_models(symbol_norm, days_ahead, period=period):
                return None

        stock_data = self.get_stock_data(symbol_norm, period=period)
        if stock_data is None:
            return None

        stock_data = self.calculate_technical_indicators(stock_data)
        # create features for prediction WITHOUT target shifting
        stock_features_df, feature_columns = self.create_features(stock_data, days_ahead, create_target=False)

        if stock_features_df.empty:
            print("Not enough data to create features for prediction.")
            return None

        latest_row = stock_features_df[feature_columns].iloc[-1:].copy()

        predictions = {}
        stored = self.models[symbol_norm]
        results = stored['results']

        for model_name, model_info in results.items():
            model = model_info['model']
            # decide scaling usage consistent with train
            if model_name in ['Linear Regression', 'SVR']:
                latest_scaled = self.scalers[symbol_norm].transform(latest_row)
                pred = model.predict(latest_scaled)[0]
            else:
                pred = model.predict(latest_row.values)[0]
            predictions[model_name] = pred

        return predictions

    def plot_predictions(self, symbol, forecast_days=30, period="2y"):
        """Plot actual vs predicted on test split (requires trained model)"""
        symbol_norm = self.normalize_symbol(symbol)
        if symbol_norm not in self.models:
            print("Model not trained for this symbol")
            return

        stock_data = self.get_stock_data(symbol_norm, period=period)
        stock_data = self.calculate_technical_indicators(stock_data)
        stock_data, feature_columns = self.create_features(stock_data, forecast_days, create_target=True)

        X = stock_data[feature_columns]
        y = stock_data['Target']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        plt.figure(figsize=(15, 10))
        results = self.models[symbol_norm]['results']

        for i, (model_name, model_info) in enumerate(results.items()):
            plt.subplot(2, 2, i + 1)
            if model_name in ['Linear Regression', 'SVR']:
                X_test_model = self.scalers[symbol_norm].transform(X_test)
                preds = model_info['model'].predict(X_test_model)
            else:
                preds = model_info['model'].predict(X_test.values)

            plt.plot(y_test.values, label='Actual', alpha=0.7)
            plt.plot(preds, label='Predicted', alpha=0.7)
            plt.title(f'{model_name} Predictions\nR2: {model_info["r2"]:.3f}, MAE: {model_info["mae"]:.3f}')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Price')

        plt.tight_layout()
        plt.show()
