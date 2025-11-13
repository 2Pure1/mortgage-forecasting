import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MortgageForecaster:
    def __init__(self, data):
        self.data = data.copy()
        self.models = {}
        self.forecasts = {}
        
    def prepare_data(self, test_size=0.2):
        """Prepare train-test split"""
        self.data = self.data.set_index('date')
        self.data = self.data.asfreq('Q')
        
        n_test = int(len(self.data) * test_size)
        self.train = self.data.iloc[:-n_test]
        self.test = self.data.iloc[-n_test:]
        
        print(f"Training period: {self.train.index.min()} to {self.train.index.max()}")
        print(f"Test period: {self.test.index.min()} to {self.test.index.max()}")
        
    def fit_sarima(self):
        """Fit SARIMA model using auto_arima"""
        print("Fitting SARIMA model...")
        
        # Check if there's enough data for auto_arima to run
        if len(self.train['total_loan_volume']) < 10: # A reasonable minimum for seasonal ARIMA
            print("Warning: Not enough training data to fit a SARIMA model. Skipping.")
            self.models['sarima'] = None # Explicitly set to None
            return None

        # Auto-select best SARIMA parameters
        model = auto_arima(
            self.train['total_loan_volume'],
            seasonal=True,
            m=4,  # quarterly data
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        self.models['sarima'] = model
        print(f"Best SARIMA parameters: {model.order}, {model.seasonal_order}")
        
        return model
    
    def fit_prophet(self):
        """Fit Facebook Prophet model"""
        print("Fitting Prophet model...")
        
        # Prepare data for Prophet
        prophet_df = self.train.reset_index()
        prophet_df = prophet_df.rename(columns={'date': 'ds', 'total_loan_volume': 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        model.fit(prophet_df)
        self.models['prophet'] = model
        
        return model
    
    def fit_ets(self):
        """Fit Exponential Smoothing model"""
        print("Fitting Exponential Smoothing model...")
        
        model = ExponentialSmoothing(
            self.train['total_loan_volume'],
            seasonal_periods=4,
            trend='add',
            seasonal='add'
        ).fit()
        
        self.models['ets'] = model
        
        return model

    def _create_features(self, df, label=None):
        """
        Creates time series features from datetime index
        """
        df = df.copy()
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        X = df[['quarter', 'year']]
        if label:
            y = df[label]
            return X, y
        return X

    def fit_xgboost(self):
        """Fit XGBoost model"""
        print("Fitting XGBoost model...")

        X_train, y_train = self._create_features(self.train, label='total_loan_volume')

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train)],
                  verbose=False)

        self.models['xgboost'] = model
        return model
    
    def generate_forecasts(self, horizon=6):
        """Generate forecasts for all models"""
        self.forecast_horizon = horizon
        
        # SARIMA forecast
        if 'sarima' in self.models:
            if self.models['sarima'] is None: return self.forecasts
            sarima_forecast = self.models['sarima'].predict(n_periods=horizon)
            self.forecasts['sarima'] = sarima_forecast
        
        # Prophet forecast
        if 'prophet' in self.models:
            future = self.models['prophet'].make_future_dataframe(periods=horizon, freq='Q')
            prophet_forecast = self.models['prophet'].predict(future)
            self.forecasts['prophet'] = prophet_forecast.tail(horizon)['yhat']
        
        # ETS forecast
        if 'ets' in self.models:
            ets_forecast = self.models['ets'].forecast(horizon)
            self.forecasts['ets'] = ets_forecast
            
        # XGBoost forecast
        if 'xgboost' in self.models:
            future_dates = pd.date_range(start=self.test.index.min(), periods=horizon, freq='Q')
            future_df = pd.DataFrame(index=future_dates)
            X_future = self._create_features(future_df)
            xgboost_forecast = self.models['xgboost'].predict(X_future)
            self.forecasts['xgboost'] = xgboost_forecast

        return self.forecasts
    
    def evaluate_models(self):
        """Evaluate model performance on test set"""
        results = {}
        
        for model_name, forecast in self.forecasts.items():
            if forecast is None: continue

            # Align forecast with test data
            test_values = self.test['total_loan_volume'].values
            forecast = forecast[:len(test_values)]
            
            mae = mean_absolute_error(test_values, forecast)
            rmse = np.sqrt(mean_squared_error(test_values, forecast))
            mape = np.mean(np.abs((test_values - forecast) / test_values)) * 100
            
            results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Accuracy': max(0, 100 - mape)  # Simplified accuracy metric
            }
            
        return pd.DataFrame(results).T
    
    def final_forecast(self, periods=8):
        """Generate final forecast using best model"""
        # Find best model based on MAPE
        if not self.models or all(v is None for v in self.models.values()):
            print("Error: No models were successfully trained. Cannot generate a final forecast.")
            return None, None, None

        evaluation = self.evaluate_models()
        best_model_name = evaluation['Accuracy'].idxmax()
        best_model = self.models[best_model_name]
        
        print(f"Best model: {best_model_name} with {evaluation.loc[best_model_name, 'Accuracy']:.2f}% accuracy")
        
        # Retrain on full data and forecast
        if best_model_name == 'sarima':
            # Retrain SARIMA on full data
            final_model = auto_arima(
                self.data['total_loan_volume'],
                seasonal=True,
                m=4,
                start_p=best_model.order[0],
                start_q=best_model.order[2],
                start_P=best_model.seasonal_order[0],
                start_Q=best_model.seasonal_order[2],
                stepwise=True
            )
            final_forecast = final_model.predict(n_periods=periods)
            
        elif best_model_name == 'prophet':
            prophet_df = self.data.reset_index().rename(columns={'date': 'ds', 'total_loan_volume': 'y'})
            best_model.fit(prophet_df)
            future = best_model.make_future_dataframe(periods=periods, freq='Q')
            forecast_df = best_model.predict(future)
            final_forecast = forecast_df.tail(periods)['yhat']

        elif best_model_name == 'xgboost':
            X_all, y_all = self._create_features(self.data, label='total_loan_volume')
            best_model.fit(X_all, y_all)
            
            last_date = self.data.index.max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=periods, freq='Q')
            future_df = pd.DataFrame(index=future_dates)
            X_future = self._create_features(future_df)
            final_forecast = best_model.predict(X_future)
            
        else:  # ETS
            final_model = ExponentialSmoothing(
                self.data['total_loan_volume'],
                seasonal_periods=4,
                trend='add',
                seasonal='add'
            ).fit()
            final_forecast = final_model.forecast(periods)
        
        # Create forecast dataframe
        last_date = self.data.index.max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=periods,
            freq='Q'
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_volume': final_forecast,
            'model': best_model_name
        })
        
        return forecast_df, evaluation, best_model_name