import os
import joblib
import pandas as pd
import numpy as np

def predict_next_30_days(df):

    model = joblib.load("models/prophet_model.pkl")

    future = model.make_future_dataframe(periods=30)

    # Add same regressors for future
    future['is_holiday'] = 0
    future['is_pongal_season'] = future['ds'].dt.month.apply(
        lambda x: 1 if x == 1 else 0
    )
    future['is_diwali_season'] = future['ds'].dt.month.apply(
        lambda x: 1 if x in [10, 11] else 0
    )
    future['temperature'] = np.random.normal(30, 5, len(future))
    future['rainfall'] = np.random.normal(5, 2, len(future))

    forecast = model.predict(future)

    # Create folder if not exists
    os.makedirs("data/processed", exist_ok=True)

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(
        "data/processed/future_predictions.xlsx",
        index=False
    )

    print("Prediction saved successfully.")

    return forecast