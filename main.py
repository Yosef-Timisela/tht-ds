# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main-title h1 {
            font-size: 64px !important;
            font-weight: 900;
            text-align: center;
            color: #0d47a1;
            margin-bottom: 2rem;
        }
        .section-title {
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .stRadio > div {
            flex-direction: row !important;
            flex-wrap: wrap;
            justify-content: center;
        }
        .stRadio label {
            margin-right: 1rem;
            font-size: 14px;
        }
        .result-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #1565c0;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load & Preprocess ---
@st.cache_data
def load_data():
    df = pd.read_csv("df_ds_final.csv")
    df['Distance_km'] = df['Distance_km'].str.replace(',', '.').astype(float)
    df['Delivery_Time_min_log'] = df['Delivery_Time_min_log'].str.replace(',', '.').astype(float)
    return df

def encode_features(df):
    label_encoders = {}
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# --- Title ---
st.markdown('<div class="section-title"><h1>🚚 Food Delivery Estimation Prediction</h1></div>', unsafe_allow_html=True)

# --- Prepare Data ---
df = load_data()
df_encoded, label_encoders = encode_features(df.copy())
X = df_encoded.drop(columns=["Order_ID", "Delivery_Time_min", "Delivery_Time_min_log"])
y = df_encoded["Delivery_Time_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
param_grid = {
    'max_depth': [3],
    'learning_rate': [0.1],
    'n_estimators': [100],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
model = xgb.XGBRegressor(random_state=42, verbosity=0)
grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# --- Layout ---
st.markdown('<div class="section-title">📝 Fill out the form below:</div>', unsafe_allow_html=True)
form_col, result_col = st.columns([2, 1])

with form_col:
    col1, col2 = st.columns(2)
    with col1:
        distance = st.slider("Distance (KM)", 0.0, 50.0, 5.0, step=0.1)
        prep_time = st.slider("Preparation Time (Min)", 0, 120, 10, step=1)
        experience = st.slider("Courier Experience (Years)", 0.0, 20.0, 2.0, step=0.5)
    with col2:
        weather = st.radio("Weather", df["Weather"].unique(), index=0)
        traffic = st.radio("Traffic Level", df["Traffic_Level"].unique(), index=0)
        time_of_day = st.radio("Time of Day", df["Time_of_Day"].unique(), index=0)
        vehicle = st.radio("Vehicle Type", df["Vehicle_Type"].unique(), index=0)

# --- Predict ---
input_data = {
    "Distance_km": distance,
    "Weather": label_encoders["Weather"].transform([weather])[0],
    "Traffic_Level": label_encoders["Traffic_Level"].transform([traffic])[0],
    "Time_of_Day": label_encoders["Time_of_Day"].transform([time_of_day])[0],
    "Vehicle_Type": label_encoders["Vehicle_Type"].transform([vehicle])[0],
    "Preparation_Time_min": prep_time,
    "Courier_Experience_yrs": experience
}
input_df = pd.DataFrame([input_data])
prediction = best_model.predict(input_df)[0]

# --- Prediction Output ---
with result_col:
    st.markdown('<div class="section-title">📦 Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="result-box">
            Estimated Delivery Time: <br><strong>{prediction:.2f} minutes</strong>
        </div>
    """, unsafe_allow_html=True)