import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Reuse preprocessing function from model_comparison.py
def preprocess_data(df):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    # Encode categorical variables
    le_country = LabelEncoder()
    le_store = LabelEncoder()
    le_product = LabelEncoder()
    
    df['country_encoded'] = le_country.fit_transform(df['country'])
    df['store_encoded'] = le_store.fit_transform(df['store'])
    df['product_encoded'] = le_product.fit_transform(df['product'])
    
    # Save the label encoders for later use in the Streamlit app
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_country, 'models/le_country.pkl')
    joblib.dump(le_store, 'models/le_store.pkl')
    joblib.dump(le_product, 'models/le_product.pkl')
    
    return df

# Define feature columns
def get_feature_columns():
    return ['year', 'month', 'day_of_week', 'day_of_month', 
            'country_encoded', 'store_encoded', 'product_encoded']

# Train the best model based on model_comparison.py results
def train_best_model(df):
    print("Preprocessing data...")
    df = preprocess_data(df)
    features = get_feature_columns()
    target = 'num_sold'
    
    # Based on model_comparison.py, we'll train XGBoost, LightGBM, and Prophet
    # and save the best performing one
    
    print("Training XGBoost model...")
    xgb_model = XGBRegressor(random_state=42)
    X = df[features]
    y = df[target]
    xgb_model.fit(X, y)
    xgb_predictions = xgb_model.predict(X)
    xgb_rmse = np.sqrt(mean_squared_error(y, xgb_predictions))
    xgb_mae = mean_absolute_error(y, xgb_predictions)
    xgb_r2 = r2_score(y, xgb_predictions)
    
    print("Training LightGBM model...")
    lgbm_model = LGBMRegressor(random_state=42)
    lgbm_model.fit(X, y)
    lgbm_predictions = lgbm_model.predict(X)
    lgbm_rmse = np.sqrt(mean_squared_error(y, lgbm_predictions))
    lgbm_mae = mean_absolute_error(y, lgbm_predictions)
    lgbm_r2 = r2_score(y, lgbm_predictions)
    
    print("Training Prophet model...")
    prophet_df = df[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(prophet_df)
    
    # Use the same dates as in the training data to ensure matching lengths
    future = prophet_df[['ds']]
    forecast = prophet_model.predict(future)
    prophet_predictions = forecast['yhat']
    prophet_rmse = np.sqrt(mean_squared_error(prophet_df['y'], prophet_predictions))
    prophet_mae = mean_absolute_error(prophet_df['y'], prophet_predictions)
    prophet_r2 = r2_score(prophet_df['y'], prophet_predictions)
    
    # Compare models and save the best one
    models = {
        'XGBoost': {'model': xgb_model, 'rmse': xgb_rmse, 'mae': xgb_mae, 'r2': xgb_r2},
        'LightGBM': {'model': lgbm_model, 'rmse': lgbm_rmse, 'mae': lgbm_mae, 'r2': lgbm_r2},
        'Prophet': {'model': prophet_model, 'rmse': prophet_rmse, 'mae': prophet_mae, 'r2': prophet_r2}
    }
    
    # Find the best model based on RMSE
    best_model_name = min(models, key=lambda x: models[x]['rmse'])
    best_model = models[best_model_name]['model']
    
    print(f"\nModel Evaluation Results:")
    for model_name, metrics in models.items():
        print(f"\n{model_name}:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R2: {metrics['r2']:.2f}")
    
    print(f"\nBest model: {best_model_name} with RMSE: {models[best_model_name]['rmse']:.2f}")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    if best_model_name in ['XGBoost', 'LightGBM']:
        joblib.dump(best_model, f'models/{best_model_name.lower()}_model.pkl')
    else:  # Prophet model
        best_model.save('models/prophet_model.json')
    
    # Save model name for reference in the Streamlit app
    with open('models/best_model.txt', 'w') as f:
        f.write(f"{best_model_name}\n")
        f.write(f"RMSE: {models[best_model_name]['rmse']:.2f}\n")
        f.write(f"MAE: {models[best_model_name]['mae']:.2f}\n")
        f.write(f"R2: {models[best_model_name]['r2']:.2f}")
    
    return best_model_name, best_model

if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_csv('train.csv')
    
    print("Training models...")
    best_model_name, best_model = train_best_model(df)
    
    print(f"\nTraining complete! The best model ({best_model_name}) has been saved to the 'models' directory.")
    print("You can now run the Streamlit app with: streamlit run app.py")