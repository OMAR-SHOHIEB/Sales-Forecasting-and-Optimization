import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Sales Forecasting App", layout="wide")

# App title and description
st.title("Sales Forecasting Application")
st.markdown("""
This application uses machine learning to predict sales based on historical data.
Upload your test data or input specific parameters to get sales predictions.
""")

# Check if models exist
if not os.path.exists('models/best_model.txt'):
    st.error("Models not found. Please run Model_Train.py first to train the models.")
    st.stop()

# Load the best model name
with open('models/best_model.txt', 'r') as f:
    model_info = f.readlines()
best_model_name = model_info[0].split(':')[0].strip()
best_r2 = model_info[3].split(': ')[1].strip()

st.info(f"Using {best_model_name} model for predictions")

# Load label encoders
le_country = joblib.load('models/le_country.pkl')
le_store = joblib.load('models/le_store.pkl')
le_product = joblib.load('models/le_product.pkl')

# Load the best model
model = joblib.load(f'models/{best_model_name.lower()}_model.pkl')

# Function to preprocess input data
def preprocess_input(df):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    # Encode categorical variables
    df['country_encoded'] = df['country'].map(lambda x: le_country.transform([x])[0] if x in le_country.classes_ else -1)
    df['store_encoded'] = df['store'].map(lambda x: le_store.transform([x])[0] if x in le_store.classes_ else -1)
    df['product_encoded'] = df['product'].map(lambda x: le_product.transform([x])[0] if x in le_product.classes_ else -1)
    
    return df

# Function to make predictions
def predict_sales(df):
    df = preprocess_input(df)
    
    features = ['year', 'month', 'day_of_week', 'day_of_month', 
                'country_encoded', 'store_encoded', 'product_encoded']
    predictions = model.predict(df[features])
    predictions = model.predict(df[features]).astype(int)
    return predictions

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Test Data", "Manual Input"])

# Tab 1: Upload test data
with tab1:
    st.header("Upload Test Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        
        if st.button("Make Predictions", key="predict_upload"):
            # Check if required columns exist
            required_cols = ['date', 'country', 'store', 'product']
            missing_cols = [col for col in required_cols if col not in test_data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Make predictions
                predictions = predict_sales(test_data)
                
                # Add predictions to the dataframe
                result_df = test_data.copy()
                result_df['predicted_sales'] = predictions
                
                # Display results
                st.subheader("Predictions")
                st.dataframe(result_df)
                
                # Download predictions
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="sales_predictions.csv",
                    mime="text/csv"
                )

# Tab 2: Manual input
with tab2:
    st.header("Manual Input")
    
    # Get unique values from label encoders
    countries = le_country.classes_
    stores = le_store.classes_
    products = le_product.classes_
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_date = st.date_input("Date", datetime.now())
            country = st.selectbox("Country", countries)
        
        with col2:
            store = st.selectbox("Store", stores)
            product = st.selectbox("Product", products)
        
        submit_button = st.form_submit_button("Predict Sales")
    
    if submit_button:
        # Create a dataframe from the inputs
        input_data = pd.DataFrame({
            'date': [prediction_date],
            'country': [country],
            'store': [store],
            'product': [product]
        })
        
        # Make prediction
        prediction = predict_sales(input_data)[0]
        
        # Display prediction with nice formatting
        st.subheader("Prediction Result")
        st.metric("Predicted Sales", f"{prediction} units")
        
        # Add some context
        st.info(f"The model predicts that {product} will sell approximately {prediction:.2f} units at {store} in {country} on {prediction_date.strftime('%Y-%m-%d')}.")

# Add information about the model
st.sidebar.header("Model Information")
st.sidebar.write(f"**Best Model:** {best_model_name}")
st.sidebar.write(f"**R² Score:** {best_r2}")
st.sidebar.write("**Features used:**")
st.sidebar.write("- Date (year, month, day of week, day of month)")
st.sidebar.write("- Country")
st.sidebar.write("- Store")
st.sidebar.write("- Product")

# Add instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. **Upload Test Data**: Upload a CSV file with date, country, store, and product columns.
2. **Manual Input**: Select specific values to get a single prediction.
3. **Download**: After predictions, you can download the results as a CSV file.
""")

# Footer
st.markdown("""---
*Sales Forecasting Application - Developed for DEPI Project*
""")