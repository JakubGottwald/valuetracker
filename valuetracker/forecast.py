import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 📈 Forecast pomocí ARIMA
# ==========================================================
def forecast_arima(history, periods=30):
    """Forecast ceny pomocí ARIMA – trénuje se jen na posledním roce dat."""
    close_prices = history['Close'].dropna()
    last_year = close_prices.last('365D')
    model = ARIMA(last_year, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# ==========================================================
# 🔮 Forecast pomocí Prophet
# ==========================================================
def forecast_prophet(history, periods=30):
    """Forecast ceny pomocí Prophet (jen poslední rok dat)."""
    df = history.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    last_year_start = df['ds'].max() - pd.DateOffset(years=1)
    df_last_year = df[df['ds'] >= last_year_start]

    model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(df_last_year)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Úprava trendu na poslední známou cenu
    last_close = df_last_year['y'].iloc[-1]
    prophet_last_known = forecast.loc[forecast['ds'] == df_last_year['ds'].iloc[-1], 'yhat'].values[0]
    offset = last_close - prophet_last_known
    forecast['yhat'] = forecast['yhat'] + offset

    return forecast

# ==========================================================
# 📉 Forecast pomocí Holt-Winters
# ==========================================================
def forecast_holt_winters(history, periods=30):
    """Forecast pomocí Holt-Winters – trénuje se jen na posledním roce dat."""
    close_prices = history['Close'].dropna()
    last_year = close_prices.last('365D')
    model = ExponentialSmoothing(last_year, trend="add", seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# ==========================================================
# 🎲 Monte Carlo simulace
# ==========================================================
def monte_carlo_simulation(history, simulations=100, days=30):
    """Monte Carlo simulace pro predikci cen."""
    close_prices = history['Close']
    returns = close_prices.pct_change().dropna()
    last_price = close_prices.iloc[-1]

    forecast_paths = []
    for _ in range(simulations):
        simulated_prices = [last_price]
        for _ in range(days):
            simulated_prices.append(simulated_prices[-1] * (1 + np.random.choice(returns)))
        forecast_paths.append(simulated_prices[1:])  # odstraníme první hodnotu
    return np.array(forecast_paths)

# ==========================================================
# 📊 Plot forecastů
# ==========================================================
def plot_forecast(history, arima_forecast, prophet_forecast, holt_forecast, days=30, ax=None):
    """
    Kreslí forecast do předaného Axes (ax).
    Pokud ax není zadán, vytvoří nový graf a vrátí ho.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    history = history.copy()
    history.index = history.index.tz_localize(None)
    prophet_forecast = prophet_forecast.copy()
    prophet_forecast['ds'] = prophet_forecast['ds'].dt.tz_localize(None)

    last_year = history.last('365D')

    # Historie
    ax.plot(last_year.index, last_year['Close'], label='Historie (poslední rok)', color='black')

    # Datum pro forecast
    future_dates = pd.date_range(start=history.index[-1], periods=days+1, freq='B')[1:]

    # ARIMA
    ax.plot(future_dates, arima_forecast, label='ARIMA', linestyle='--', color='blue')

    # Prophet
    prophet_future = prophet_forecast[prophet_forecast['ds'] > history.index[-1]].head(days)
    ax.plot(prophet_future['ds'], prophet_future['yhat'], label='Prophet', linestyle='--', color='green')

    # Holt-Winters
    ax.plot(future_dates, holt_forecast, label='Holt-Winters', linestyle='--', color='orange')

    ax.set_title("📈 Forecast ceny akcie (ARIMA, Prophet & Holt-Winters)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Cena (USD)")
    ax.legend()
    ax.grid(True)

    if created_fig:
        return fig, ax
    else:
        return ax

# ==========================================================
# 📊 Plot Monte Carlo simulace
# ==========================================================
def plot_monte_carlo(history, mc_simulation, days=30, ax=None):
    """
    Kreslí Monte Carlo simulace do předaného Axes (ax).
    Pokud ax není zadán, vytvoří nový graf a vrátí ho.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    history = history.copy()
    history.index = history.index.tz_localize(None)

    last_year = history.last('365D')

    future_dates = pd.date_range(start=history.index[-1], periods=days+1, freq='B')[1:]

    # Historie
    ax.plot(last_year.index, last_year['Close'], color='black', label='Historie (poslední rok)')

    # Monte Carlo simulace
    for i in range(mc_simulation.shape[0]):
        ax.plot(future_dates, mc_simulation[i], color='gray', alpha=0.1)

    ax.set_title("📈 Monte Carlo simulace cen akcie")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Cena (USD)")
    ax.legend()
    ax.grid(True)

    if created_fig:
        return fig, ax
    else:
        return ax

# ==========================================================
# 📢 Hodnocení forecastu
# ==========================================================
def evaluate_forecast(arima_forecast):
    """Shrnutí forecastu – pokud průměr roste, označíme bullish."""
    start_price = arima_forecast.iloc[0]
    end_price = arima_forecast.iloc[-1]
    if end_price > start_price * 1.03:
        return "🚀 Forecast ukazuje růstový trend."
    elif end_price < start_price * 0.97:
        return "📉 Forecast ukazuje poklesový trend."
    else:
        return "⚖️ Forecast naznačuje spíše stabilní vývoj."
