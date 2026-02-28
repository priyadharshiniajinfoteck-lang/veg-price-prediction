from data_preprocessing import load_data
from add_external_features import add_holiday_feature, add_festival_feature, add_weather_feature
from train_prophet import train_model
from predict_future import predict_next_30_days

file_path = "Supermarket_Fruits_Vegetables_1000_Rows.xlsx"

df = load_data(file_path)

df = add_holiday_feature(df)
df = add_festival_feature(df)
df = add_weather_feature(df)

model = train_model(df)

forecast = predict_next_30_days(df)

from weather_api import get_weather_data

df = load_data(file_path)

weather_df = get_weather_data(df['ds'].min(), df['ds'].max())

df = df.merge(weather_df, on='ds', how='left')

df.fillna(method='ffill', inplace=True)

from train_per_product import train_models_per_product

df = pd.read_excel(file_path)

train_models_per_product(df)