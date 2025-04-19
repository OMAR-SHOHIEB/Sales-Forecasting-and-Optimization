import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet

# Load and preprocess data
def preprocess_data(df):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    # Encode categorical variables
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['country'])
    df['store_encoded'] = le.fit_transform(df['store'])
    df['product_encoded'] = le.fit_transform(df['product'])
    
    return df

# Define feature columns
def get_feature_columns():
    return ['year', 'month', 'day_of_week', 'day_of_month', 
            'country_encoded', 'store_encoded', 'product_encoded']

# Train and evaluate models
def evaluate_models(df):
    df = preprocess_data(df)
    features = get_feature_columns()
    target = 'num_sold'
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_idx, test_idx in tscv.split(df):
            X_train = df.iloc[train_idx][features]
            y_train = df.iloc[train_idx][target]
            X_test = df.iloc[test_idx][features]
            y_test = df.iloc[test_idx][target]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
        
        results[name] = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores),
            'R2': np.mean(r2_scores)
        }
    
    # Prophet model evaluation
    prophet_df = df[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_rmse = []
    prophet_mae = []
    prophet_r2 = []
    
    for train_idx, test_idx in tscv.split(prophet_df):
        train_data = prophet_df.iloc[train_idx]
        test_data = prophet_df.iloc[test_idx]
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(train_data)
        
        future_dates = test_data[['ds']]
        forecast = model.predict(future_dates)
        
        rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
        mae = mean_absolute_error(test_data['y'], forecast['yhat'])
        r2 = r2_score(test_data['y'], forecast['yhat'])
        
        prophet_r2.append(r2)
        prophet_rmse.append(rmse)
        prophet_mae.append(mae)
    
    results['Prophet'] = {
        'RMSE': np.mean(prophet_rmse),
        'MAE': np.mean(prophet_mae),
        'R2': np.mean(prophet_r2)
    }
    
    return results

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('train.csv')
    
    # Evaluate models
    results = evaluate_models(df)
    
    # Print results
    print("\nModel Evaluation Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"R2: {metrics['R2']:.2f}")