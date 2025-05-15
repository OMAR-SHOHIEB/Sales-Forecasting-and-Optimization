
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 🎨 CSS to increase font size
# ==============================
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🔄 Load and preprocess data
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_name()
    df = df[df["num_sold"] < 950]  # remove outliers
    return df

df = load_data()

# ==============================
# 🔀 Navigation
# ==============================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Preprocessing", "Visualization", "EDA", "Model"])

# ==============================
# 🧹 Preprocessing Section
# ==============================
if option == "Preprocessing":
    st.title("🧹 Data Preprocessing")

    st.subheader("1️⃣ Load Dataset")
    st.write("Shape of the data:", df)

    st.subheader("2️⃣ Convert 'date' to datetime")
    st.code("df['date'] = pd.to_datetime(df['date'])")

    st.subheader("3️⃣ Extract Time Features")
    st.markdown("""
    - `df['year'] = df['date'].dt.year`
    - `df['month'] = df['date'].dt.month`
    - `df['day'] = df['date'].dt.day`
    - `df['day_of_week'] = df['date'].dt.day_name()`
    """)

    st.subheader("4️⃣ Remove Outliers")
    st.code("df = df[df['num_sold'] < 950]")

    st.subheader("5️⃣ Check Missing Values")
    st.write(df.isna().sum())

    st.subheader("6️⃣ Data Types")
    st.write(df.dtypes)

    st.subheader("7️⃣ Descriptive Statistics")
    st.write(df.describe())

# ==============================
# 📊 Visualization Section
# ==============================
elif option == "Visualization":
    st.title("📊 All Visualizations")

    st.subheader("1️⃣ Sales Over Time")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["num_sold"], linewidth=0.3)
    ax.set_title("Sales Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("2️⃣ Sales by Year")
    sales_by_year = df.groupby("year")["num_sold"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(sales_by_year["year"], sales_by_year["num_sold"], marker='o')
    ax.set_title("Total Sales by Year")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("3️⃣ Sales by Month")
    sales_by_month = df.groupby("month")["num_sold"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(sales_by_month["month"], sales_by_month["num_sold"], marker='o')
    ax.set_title("Total Sales by Month")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("4️⃣ Sales by Day of Month")
    sales_by_day = df.groupby("day")["num_sold"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(sales_by_day["day"], sales_by_day["num_sold"])
    ax.set_title("Total Sales by Day of Month")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("5️⃣ Full Time Series with Zoomed Lines")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["num_sold"], linewidth=0.1)
    ax.set_title("Zoomed Time Series")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("6️⃣ Sales by Day of Week and Country")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='day_of_week', y='num_sold', hue='country', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Sales by Day and Country")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    for year in sorted(df["year"].unique()):
        st.subheader(f"7️⃣ Sales Distribution in {year}")
        data_year = df[df["year"] == year]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data_year["date"], data_year["num_sold"], linewidth=0.1)
        ax.set_title(f"Sales in {year}")
        st.pyplot(fig)

    st.subheader("8️⃣ Total Sales by Country")
    sales_by_country = df.groupby("country")["num_sold"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.bar(sales_by_country["country"], sales_by_country["num_sold"], color=["blue", "green", "red"])
    ax.set_title("Total Sales by Country")
    st.pyplot(fig)

    st.subheader("9️⃣ Barplot - Country Sales")
    fig, ax = plt.subplots()
    sns.barplot(x='country', y='num_sold', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Sales by Country")
    st.pyplot(fig)

    st.subheader("🔟 Boxplot - Country Sales Spread")
    fig, ax = plt.subplots()
    sns.boxplot(x='country', y='num_sold', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Sales Spread by Country")
    st.pyplot(fig)

    st.subheader("1️⃣1️⃣ Yearly Growth by Country")
    df_yearly = df.groupby(['year', 'country'])['num_sold'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(x='year', y='num_sold', hue='country', data=df_yearly, marker='o', ax=ax)
    ax.set_title("Yearly Sales by Country")
    st.pyplot(fig)

    st.subheader("1️⃣2️⃣ Store Sales by Country")
    fig, ax = plt.subplots()
    sns.barplot(x='country', y='num_sold', hue='store', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Store Sales by Country")
    st.pyplot(fig)

    st.subheader("1️⃣3️⃣ Total Sales by Store")
    store_sales = df.groupby("store")["num_sold"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.bar(store_sales["store"], store_sales["num_sold"], color=["blue", "green", "red"])
    ax.set_title("Sales by Store")
    st.pyplot(fig)

    st.subheader("1️⃣4️⃣ Product Sales per Store")
    fig, ax = plt.subplots()
    sns.barplot(x='store', y='num_sold', hue='product', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Product Sales in Stores")
    st.pyplot(fig)

# ==============================
# 📈 EDA Section
# ==============================
elif option == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    # Correlation Heatmap
    st.subheader("1️⃣ Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlation Between Numeric Features")
    st.pyplot(fig)
    st.markdown("🧠 **Insight**: Weak correlations overall, but time-related features like 'year' and 'month' show minor trends with sales.")

    # Country Sales Summary
    st.subheader("2️⃣ Sales Summary by Country")
    summary = df.groupby("country")["num_sold"].agg(["mean", "median", "std", "sum", "count"]).reset_index()
    st.dataframe(summary)

    # Outlier Detection
    st.subheader("3️⃣ Outlier Detection in 'num_sold'")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["num_sold"], ax=ax)
    ax.set_title("Boxplot of Sales")
    st.pyplot(fig)
    st.markdown("🔎 **Observation**: Before preprocessing, some values exceeded 950. These were filtered out.")

    # Category Distributions
    st.subheader("4️⃣ Product Distribution by Country")
    fig, ax = plt.subplots()
    sns.countplot(x="product", hue="country", data=df, palette="pastel", ax=ax)
    ax.set_title("Product Type Counts by Country")
    st.pyplot(fig)

    st.subheader("5️⃣ Store Distribution by Country")
    fig, ax = plt.subplots()
    sns.countplot(x="store", hue="country", data=df, palette="Set2", ax=ax)
    ax.set_title("Store Type Counts by Country")
    st.pyplot(fig)

    st.success("✅ EDA section complete. You can now explore preprocessing and visualizations from the sidebar.")
    
elif option == "Model":
    exec(open("app.py", encoding="utf-8").read())