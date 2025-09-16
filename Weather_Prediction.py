import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Hourly
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px

def get_city_coordinates(city_name):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        st.error(f"City '{city_name}' not found! Using Delhi as default.")
        return 28.6139, 77.2090  


def fetch_weather_data(city, days):
    lat, lon = get_city_coordinates(city)
    point = Point(lat, lon)
    end = datetime.now()
    start = end - timedelta(days=days)

    data = Hourly(point, start, end).fetch().reset_index()

    data = data[['time', 'temp', 'rhum', 'pres', 'wspd']]
    data.columns = ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed']
    data = data.dropna()
    return data


def prepare_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    features = ['hour', 'day_of_week', 'month']
    X = df[features]

    # Multi-output target
    Y = df[['temperature', 'humidity', 'pressure', 'wind_speed']]
    return X, Y


def predict_next_24_hours(df):
    X, Y = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, Y)   

    # Predict for next 24 hours
    now = datetime.now()
    hours_ahead = [now + timedelta(hours=i) for i in range(1, 25)]
    pred_df = pd.DataFrame({'timestamp': hours_ahead})
    pred_df['hour'] = pred_df['timestamp'].dt.hour
    pred_df['day_of_week'] = pred_df['timestamp'].dt.dayofweek
    pred_df['month'] = pred_df['timestamp'].dt.month

    X_pred_scaled = scaler.transform(pred_df[['hour', 'day_of_week', 'month']])
    preds = model.predict(X_pred_scaled)

    pred_df[['predicted_temp', 'predicted_humidity',
             'predicted_pressure', 'predicted_wind']] = preds

    return pred_df


st.set_page_config(page_title="Weather Prediction using AI", page_icon="🌦", layout="wide")
st.title("Multi-Parameter Weather Prediction AI")

city_input = st.text_input("Enter City Name", "Delhi")

if st.button("Predict Next 24 Hours"):
    with st.spinner(f"Fetching data for {city_input}"):
        # 7 days for training
        training_data = fetch_weather_data(city_input, 7)
        predicted_data = predict_next_24_hours(training_data)
        # 2 days for plotting
        historical_data = fetch_weather_data(city_input, 2)

        # Temperature Chart
        combined_temp = pd.concat([
            historical_data[['timestamp', 'temperature']].rename(columns={'temperature': 'value'}).assign(type="Historical"),
            predicted_data[['timestamp', 'predicted_temp']].rename(columns={'predicted_temp': 'value'}).assign(type="Predicted")
        ])
        fig_temp = px.line(combined_temp, x='timestamp', y='value', color='type',
                        title=f"Temperature Trend for {city_input}")
        fig_temp.update_traces(hovertemplate="Time=%{x}<br>Temperature=%{y} °C")
        fig_temp.update_layout(
            template="plotly_white",
            yaxis_title="Temperature (°C)",
            xaxis_title="Time",
            height=300
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # Humidity Chart
        combined_humidity = pd.concat([
            historical_data[['timestamp', 'humidity']]
                .rename(columns={'humidity': 'value'})
                .assign(type="Historical"),
            predicted_data[['timestamp', 'predicted_humidity']]
                .rename(columns={'predicted_humidity': 'value'})
                .assign(type="Predicted")
        ])
        fig_humidity = px.line(
            combined_humidity,
            x="timestamp",
            y="value",
            color="type",
            title=f"Humidity Trend for {city_input}",
            markers=True
        )
        fig_humidity.update_traces(line=dict(width=2), hovertemplate="Time=%{x}<br>Humidity=%{y}%")
        fig_humidity.update_layout(
            template="plotly_white",
            yaxis_title="Humidity (%)",
            xaxis_title="Time",
            height=300
        )
        st.plotly_chart(fig_humidity, use_container_width=True)

        # Pressure Chart
        combined_pressure = pd.concat([
            historical_data[['timestamp', 'pressure']].rename(columns={'pressure': 'value'}).assign(type="Historical"),
            predicted_data[['timestamp', 'predicted_pressure']].rename(columns={'predicted_pressure': 'value'}).assign(type="Predicted")
        ])
        fig_pressure = px.line(combined_pressure, x='timestamp', y='value', color='type',
                            title=f"Pressure Trend for {city_input}")
        fig_pressure.update_traces(hovertemplate="Time=%{x}<br>Pressure=%{y} hPa")
        fig_pressure.update_layout(
            template="plotly_white",
            yaxis_title="Pressure (hPa)",
            xaxis_title="Time",
            height=300
        )
        st.plotly_chart(fig_pressure, use_container_width=True)

        # Wind Speed Chart
        combined_wind = pd.concat([
            historical_data[['timestamp', 'wind_speed']].rename(columns={'wind_speed': 'value'}).assign(type="Historical"),
            predicted_data[['timestamp', 'predicted_wind']].rename(columns={'predicted_wind': 'value'}).assign(type="Predicted")
        ])
        fig_wind = px.bar(combined_wind, x='timestamp', y='value', color='type',
                        barmode="group", title=f"Wind Speed for {city_input}")
        fig_wind.update_traces(hovertemplate="Time=%{x}<br>Wind Speed=%{y} km/h")
        fig_wind.update_layout(
            template="plotly_white",
            yaxis_title="Wind Speed (km/h)",  
            xaxis_title="Time",
            height=300
        )
        st.plotly_chart(fig_wind, use_container_width=True)


        st.subheader("Predicted Hourly Weather Parameters for Next 24 Hours")
        st.dataframe(predicted_data[['Time', 'Predicted Tempearture',
                                    'Predicted Humidity', 'Predicted_Pressure', 'Predicted Wind']])
