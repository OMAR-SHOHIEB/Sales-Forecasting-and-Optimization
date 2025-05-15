import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime


st.title("Sales Forecasting Application")
st.markdown("""
This application uses machine learning to predict sales based on historical data.
Upload your test data or input specific parameters to get sales predictions.
""")

# تأكد من وجود ملفات النماذج
if not os.path.exists('models/best_model.txt'):
    st.error("Models not found. Please run Model_Train.py first to train the models.")
    st.stop()

# تحميل اسم أفضل نموذج
with open('models/best_model.txt', 'r') as f:
    model_info = f.readlines()

best_model_name = model_info[0].split(':')[0].strip()
best_r2 = model_info[3].split(': ')[1].strip()
st.info(f"Using {best_model_name} model for predictions")

# تحميل label encoders والنموذج
le_country = joblib.load('models/le_country.pkl')
le_store = joblib.load('models/le_store.pkl')
le_product = joblib.load('models/le_product.pkl')
model = joblib.load(f'models/{best_model_name.lower()}_model.pkl')

# دوال المعالجة والتنبؤ
def preprocess_input(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['country_encoded'] = df['country'].map(lambda x: le_country.transform([x])[0] if x in le_country.classes_ else -1)
    df['store_encoded'] = df['store'].map(lambda x: le_store.transform([x])[0] if x in le_store.classes_ else -1)
    df['product_encoded'] = df['product'].map(lambda x: le_product.transform([x])[0] if x in le_product.classes_ else -1)
    return df

def predict_sales(df):
    df = preprocess_input(df)
    features = ['year', 'month', 'day_of_week', 'day_of_month', 
                'country_encoded', 'store_encoded', 'product_encoded']
    return model.predict(df[features]).astype(int)

# Tabs
tab1, tab2 = st.tabs(["Upload Test Data", "Manual Input"])

with tab1:
    st.header("Upload Test Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        required_cols = ['date', 'country', 'store', 'product']
        if all(col in test_data.columns for col in required_cols):
            predictions = predict_sales(test_data)
            result_df = test_data.copy()
            result_df['predicted_sales'] = predictions
            st.subheader("Predictions")
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="sales_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Missing required columns: {', '.join(set(required_cols) - set(test_data.columns))}")

with tab2:
    st.header("Manual Input")
    countries = le_country.classes_
    stores = le_store.classes_
    products = le_product.classes_

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
        input_data = pd.DataFrame({
            'date': [prediction_date],
            'country': [country],
            'store': [store],
            'product': [product]
        })
        prediction = predict_sales(input_data)[0]
        st.subheader("Prediction Result")
        st.metric("Predicted Sales", f"{prediction} units")
