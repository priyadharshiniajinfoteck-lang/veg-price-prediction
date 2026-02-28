import os
import joblib
from prophet import Prophet

def train_model(df):

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )

    model.add_regressor('temperature')
    model.add_regressor('humidity')
    model.add_regressor('rainfall')
    print(df.columns)
    model.fit(df)

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/prophet_model.pkl")

    return model