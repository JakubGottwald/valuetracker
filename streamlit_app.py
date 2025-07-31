# 📈 STOCK ANALYSIS MODULE
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from valuetracker.data import get_stock_data
from valuetracker.sp500_loader import load_sp500
from valuetracker.technicals import evaluate_technicals
from valuetracker.valuation_models import evaluate_academic_models
from valuetracker.forecast import (
    forecast_arima,
    forecast_prophet,
    forecast_holt_winters,
    monte_carlo_simulation,
    plot_forecast,
    plot_monte_carlo,
    evaluate_forecast
)
from valuetracker.econometrics import evaluate_econometrics
from valuetracker.risk_models import evaluate_risk
from valuetracker.intrinsic_value import evaluate_intrinsic_value

# ✅ grading funkce (zůstává stejná)
def grade_from_score(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    elif score >= 50:
        return "E"
    else:
        return "F"

def stock_analysis_page():
    st.header("📈 Analýza vybrané akcie")

    # ✅ načteme seznam společností z S&P 500
    sp500_df = load_sp500()
    all_options = [f"{row['Name']} ({row['Symbol']})" for _, row in sp500_df.iterrows()]

    # 🎛 UI komponenty
    selection = st.selectbox("Vyber akcii:", all_options)
    period = st.selectbox("Období grafu:", ['6mo', '1y', '3y', 'ytd', 'max'], index=1)
    forecast_days = st.selectbox("Forecast:", [0, 30, 90, 180], index=0)

    if st.button("🔍 Spustit analýzu"):
        name = selection.split(" (")[0]
        symbol = selection.split("(")[-1].replace(")", "")

        st.info(f"📥 Stahuji data pro **{name} ({symbol})**…")
        data = get_stock_data(symbol, period="max")

        # ✅ rozbalíme data
        info = data['info']
        history = data['history']
        financials = data['financials']
        balance_sheet = data['balance_sheet']
        cashflow_statement = data['cashflow_statement']

        st.success("✅ Data načtena!")

        # 🏷 základní info
        st.subheader("📊 Základní informace")
        st.markdown(f"**Název:** {info.get('shortName', 'N/A')}")
        st.markdown(f"**Sektor:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Zaměstnanci:** {info.get('fullTimeEmployees', 'N/A')}")

        # 📈 graf closing price
        st.subheader("📈 Vývoj ceny akcie")
        history.index = history.index.tz_localize(None)
        if period == '6mo':
            plot_data = history.loc[history.index >= pd.Timestamp.today() - pd.DateOffset(months=6)]
        elif period == '1y':
            plot_data = history.loc[history.index >= pd.Timestamp.today() - pd.DateOffset(years=1)]
        elif period == '3y':
            plot_data = history.loc[history.index >= pd.Timestamp.today() - pd.DateOffset(years=3)]
        elif period == 'ytd':
            current_year = pd.Timestamp.today().year
            plot_data = history[history.index >= f'{current_year}-01-01']
        else:
            plot_data = history

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(plot_data.index, plot_data['Close'], label='Close Price', color='blue')
        ax.set_title(f"📈 Vývoj ceny {name} ({symbol})")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Cena (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # ✅ posledních 5 closing prices
        st.subheader("📊 Posledních 5 closing prices")
        st.dataframe(history['Close'].tail(5).iloc[::-1])

        # ✅ Technická analýza
        st.subheader("📊 Technická analýza")
        ta = evaluate_technicals(history)
        for k, v in ta['signals'].items():
            st.write(f"{k}: {v}")
        st.write(f"➡️ TA hodnocení: **{ta['overall']}**")

        # 📚 Akademické modely
        st.subheader("📚 Akademické modely")
        academic = evaluate_academic_models(history)
        for model_name in ["Sharpe Ratio", "Beta", "Treynor Ratio", "Jensen's Alpha", "Black–Scholes (demo call)"]:
            val = academic[model_name]
            comment = academic["Komentáře"].get(model_name, "")
            st.write(f"{model_name}: {val} – {comment}")
        st.write(f"🎯 Závěr: **{academic['Hodnocení']}**")

        # 📊 EKONOMETRIE
        st.subheader("📊 Ekonometrická analýza")
        econ = evaluate_econometrics(history)
        for section, results in econ.items():
            if isinstance(results, dict):
                st.markdown(f"**{section}**")
                for k, v in results.items():
                    st.write(f"{k}: {v}")
        st.write(f"📢 Hodnocení: **{econ['Hodnocení']}**")

        # ⚠️ Riziková analýza
        st.subheader("⚠️ Riziková analýza")
        risk = evaluate_risk(history)
        for k, v in risk.items():
            if k not in ["Komentáře", "Skóre rizika"]:
                st.write(f"{k}: {v}")
        for c in risk["Komentáře"]:
            st.write(f"💬 {c}")

        # 💰 Intrinsic Value
        st.subheader("💰 Intrinsic Value (DCF, DDM)")
        intrinsic = evaluate_intrinsic_value(info, financials, history, cashflow_statement)
        for c in intrinsic["Komentáře"]:
            st.write(f"💬 {c}")
        st.write(f"➡️ Status: **{intrinsic['Status']}**")

        # ✅ Finanční výkazy (poslední rok)
        st.subheader("📊 Finanční výkazy (poslední rok)")
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
        latest_financials = financials.iloc[:, 0] if not financials.empty else pd.Series()
        latest_cashflow = cashflow_statement.iloc[:, 0] if not cashflow_statement.empty else pd.Series()

        if not latest_balance.empty:
            st.markdown("**🏦 Rozvaha**")
            st.write(latest_balance.map(lambda x: f"{x:,.0f}"))
        else:
            st.warning("❌ Rozvaha nebyla nalezena.")

        if not latest_financials.empty:
            st.markdown("**📈 Výkaz zisku a ztrát**")
            st.write(latest_financials.map(lambda x: f"{x:,.0f}"))
        else:
            st.warning("❌ Výkaz zisku a ztrát nebyl nalezen.")

        if not latest_cashflow.empty:
            st.markdown("**💵 Cashflow**")
            st.write(latest_cashflow.map(lambda x: f"{x:,.0f}"))
        else:
            st.warning("❌ Cashflow nebyl nalezen.")

                # ✅ Poměrové ukazatele
        st.subheader("📊 Poměrové ukazatele")
        try:
            # 📈 Výpočet základních ukazatelů
            roa = latest_financials["Net Income"] / latest_balance["Total Assets"] if latest_balance.get("Total Assets", 0) != 0 else None
            roe = latest_financials["Net Income"] / (latest_balance.get("Total Assets", 0) - latest_balance.get("Total Liabilities", 0)) if (latest_balance.get("Total Assets", 0) - latest_balance.get("Total Liabilities", 0)) != 0 else None
            current_ratio = latest_balance["Current Assets"] / latest_balance["Current Liabilities"] if latest_balance.get("Current Liabilities", 0) != 0 else None
            quick_ratio = (latest_balance["Current Assets"] - latest_balance["Inventory"]) / latest_balance["Current Liabilities"] if latest_balance.get("Current Liabilities", 0) != 0 else None
            debt_to_equity = latest_balance.get("Total Liabilities", 0) / latest_balance.get("Ordinary Shares Number", 1)
            gross_margin = latest_financials["Gross Profit"] / latest_financials["Total Revenue"] if latest_financials.get("Total Revenue", 0) != 0 else None

            ratios = {
                "ROA": roa,
                "ROE": roe,
                "Current Ratio": current_ratio,
                "Quick Ratio": quick_ratio,
                "Debt to Equity": debt_to_equity,
                "Gross Margin": gross_margin
            }

            # 📌 Benchmark hodnoty – podle akademických a praktických standardů
            benchmarks = {
                "ROA": 0.05,              # > 5 % dobré
                "ROE": 0.10,              # > 10 % dobré
                "Current Ratio": (1.5, 3),# optimální 1.5–3
                "Quick Ratio": 1.0,       # > 1 dobré
                "Debt to Equity": 1.0,    # < 1 dobré
                "Gross Margin": 0.40      # > 40 % silná firma
            }

            comparison_table = []
            score_from_ratios = 0

            for ratio_name, value in ratios.items():
                if value is None:
                    comparison_table.append({
                        "Ukazatel": ratio_name,
                        "Hodnota": "N/A",
                        "Benchmark": "N/A",
                        "Hodnocení": "⚠️"
                    })
                    continue

                emoji = "⚖️"
                benchmark_display = benchmarks[ratio_name]

                if ratio_name == "Current Ratio":
                    # zvláštní případ: chceme, aby bylo mezi 1.5 a 3
                    if 1.5 <= value <= 3:
                        emoji = "✅"
                        score_from_ratios += 2
                    elif value < 1.0:
                        emoji = "❌"  # velmi nízká likvidita
                    else:
                        emoji = "⚖️"  # příliš vysoké, ale ne katastrofa
                elif ratio_name == "Debt to Equity":
                    if value < benchmarks["Debt to Equity"]:
                        emoji = "✅"
                        score_from_ratios += 2
                    elif value > 2:
                        emoji = "❌"
                    else:
                        emoji = "⚖️"
                else:
                    # pro všechny ostatní (ROA, ROE, Quick, Gross Margin)
                    if value > benchmarks[ratio_name]:
                        emoji = "✅"
                        score_from_ratios += 2
                    elif ratio_name in ["ROA", "ROE"] and value < 0:
                        emoji = "❌"
                    else:
                        emoji = "⚖️"

                # ✅ Přidáme do tabulky
                comparison_table.append({
                    "Ukazatel": ratio_name,
                    "Hodnota": round(value, 2),
                    "Benchmark": benchmark_display if not isinstance(benchmark_display, tuple) else f"{benchmark_display[0]}–{benchmark_display[1]}",
                    "Hodnocení": emoji
                })

            # 📊 Zobrazení tabulky s porovnáním
            st.markdown("### 📊 Porovnání s benchmarky")
            st.dataframe(pd.DataFrame(comparison_table))

        except Exception as e:
            st.error(f"❌ Chyba při výpočtu poměrových ukazatelů: {e}")

        # 🔮 Forecast
        if forecast_days > 0:
            st.subheader(f"🔮 Forecast na {forecast_days} dní")
            arima_forecast = forecast_arima(history, periods=forecast_days)
            prophet_forecast = forecast_prophet(history, periods=forecast_days)
            holt_forecast = forecast_holt_winters(history, periods=forecast_days)

            # graf forecastů
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            plot_forecast(history, arima_forecast, prophet_forecast, holt_forecast, days=forecast_days, ax=ax1)
            st.pyplot(fig1)

            # Monte Carlo simulace
            mc_simulation = monte_carlo_simulation(history, simulations=200, days=forecast_days)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            plot_monte_carlo(history, mc_simulation, days=forecast_days, ax=ax2)
            st.pyplot(fig2)

            comment = evaluate_forecast(arima_forecast)
            st.info(f"📢 {comment}")

        # 🏆 Celkové hodnocení akcie
        st.subheader("🏆 Celkové hodnocení akcie")
        total_score = 0
        if "bullish" in ta['overall'].lower():
            total_score += 18
        elif "neutral" in ta['overall'].lower():
            total_score += 12
        else:
            total_score += 6

        if "POZITIVNĚ" in academic['Hodnocení']:
            total_score += 18
        elif "NEUTRÁLNĚ" in academic['Hodnocení']:
            total_score += 12
        else:
            total_score += 6

        if "✅" in econ['Hodnocení']:
            total_score += 15
        else:
            total_score += 10

        total_score += (risk['Skóre rizika'] * 0.4)

        if "PODHODNOCENÁ" in intrinsic['Status']:
            total_score += 8
        elif "NADHODNOCENÁ" in intrinsic['Status']:
            total_score -= 8

        final_score = max(0, round(total_score, 1))
        grade = grade_from_score(final_score)

        st.success(f"✅ Skóre akcie: **{final_score}/100** – Známka: **{grade}**")


# Portfolio builder
# 📄 streamlit_app.py (přidat pod stock_analysis_page)

import numpy as np
import yfinance as yf

def portfolio_builder_page():
    st.header("💼 Portfolio Builder")

    # ✅ Načteme seznam společností z S&P 500
    sp500_df = load_sp500()
    tickers = st.multiselect("📊 Vyber akcie do portfolia:", sp500_df["Symbol"].tolist(), default=["AAPL", "MSFT"])

    strategy = st.selectbox("📈 Vyber strategii:", ["📊 Uživatelské váhy", "⚖️ Equal Weight"])

    weights = {}
    if strategy == "📊 Uživatelské váhy":
        st.write("✏️ Zadej váhy jednotlivých akcií (součet = 1.0)")
        for ticker in tickers:
            weights[ticker] = st.number_input(f"Váha pro {ticker}", min_value=0.0, max_value=1.0, value=round(1/len(tickers),2), step=0.05)
    else:
        # equal weight automaticky
        for ticker in tickers:
            weights[ticker] = 1/len(tickers) if len(tickers) > 0 else 0

    if st.button("📈 Spočítat portfolio"):
        if not tickers:
            st.warning("⚠️ Vyber aspoň jednu akcii.")
            return

        # ✅ stáhneme data z Yahoo Finance
        data = yf.download(selected_stocks, period="3y", group_by="ticker", auto_adjust=True)

        if isinstance(raw_data.columns, pd.MultiIndex):
            data = pd.concat([raw_data[ticker]["Close"].rename(ticker) for ticker in selected_stocks], axis=1)
        else:
            data = raw_data[["Close"]].rename(columns={"Close": selected_stocks[0]})

        # ✅ spočítáme výnosy
        returns = data.pct_change().dropna()
        portfolio_return = (returns * list(weights.values())).sum(axis=1)

        # ✅ základní metriky
        avg_return = portfolio_return.mean() * 252
        volatility = portfolio_return.std() * np.sqrt(252)
        sharpe = avg_return / volatility if volatility != 0 else 0
        cumulative = (1 + portfolio_return).cumprod()

        # ✅ výpis metrik
        st.subheader("📊 Portfolio metriky")
        st.write(f"📈 Očekávaný roční výnos: **{avg_return:.2%}**")
        st.write(f"⚠️ Volatilita: **{volatility:.2%}**")
        st.write(f"📊 Sharpe ratio: **{sharpe:.2f}**")

        # ✅ graf vývoje portfolia
        st.subheader("📉 Vývoj portfolia")
        fig, ax = plt.subplots(figsize=(10,5))
        cumulative.plot(ax=ax, color="blue", label="Portfolio")
        ax.set_title("Vývoj hodnoty portfolia")
        ax.set_ylabel("Hodnota (start=1)")
        ax.legend()
        st.pyplot(fig)

        # ✅ pie chart složení portfolia
        st.subheader("🥧 Složení portfolia")
        fig2, ax2 = plt.subplots()
        ax2.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
        ax2.set_title(f"Portfolio složení – strategie {strategy}")
        st.pyplot(fig2)

def calculate_max_drawdown(cumulative_returns):
    """Spočítá největší propad portfolia."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()
    return max_dd

def calculate_sortino_ratio(portfolio_return, risk_free=0):
    """Sortino ratio – penalizuje jen záporné výnosy."""
    downside = portfolio_return[portfolio_return < 0].std()
    avg_return = portfolio_return.mean() * 252
    return (avg_return - risk_free) / (downside * np.sqrt(252)) if downside != 0 else 0

def calculate_beta(portfolio_return, benchmark_return):
    """Beta vůči benchmarku (S&P 500)."""
    covariance = np.cov(portfolio_return, benchmark_return)[0][1]
    market_var = np.var(benchmark_return)
    return covariance / market_var if market_var != 0 else 0

def portfolio_builder_page():
    st.header("💼 Portfolio Builder")

    # ✅ Načteme seznam společností z S&P 500
    sp500_df = load_sp500()
    tickers = st.multiselect("📊 Vyber akcie do portfolia:", sp500_df["Symbol"].tolist(), default=["AAPL", "MSFT"])

    strategy = st.selectbox("📈 Vyber strategii:", ["📊 Uživatelské váhy", "⚖️ Equal Weight"])

    weights = {}
    if strategy == "📊 Uživatelské váhy":
        st.write("✏️ Zadej váhy jednotlivých akcií (součet = 1.0)")
        for ticker in tickers:
            weights[ticker] = st.number_input(f"Váha pro {ticker}", min_value=0.0, max_value=1.0, value=round(1/len(tickers),2), step=0.05)
    else:
        # equal weight automaticky
        for ticker in tickers:
            weights[ticker] = 1/len(tickers) if len(tickers) > 0 else 0

    if st.button("📈 Spočítat portfolio"):
        if not tickers:
            st.warning("⚠️ Vyber aspoň jednu akcii.")
            return

        # ✅ stáhneme data z Yahoo Finance (portfolio + S&P 500)
        data = yf.download(tickers + ["^GSPC"], period="1y")["Adj Close"]
        sp500 = data["^GSPC"]
        data = data.drop(columns="^GSPC")

        # ✅ výpočty výnosů
        returns = data.pct_change().dropna()
        sp500_returns = sp500.pct_change().dropna()

        # ✅ portfolio výnos
        portfolio_return = (returns * list(weights.values())).sum(axis=1)
        cumulative = (1 + portfolio_return).cumprod()

        # ✅ metriky
        avg_return = portfolio_return.mean() * 252
        volatility = portfolio_return.std() * np.sqrt(252)
        sharpe = avg_return / volatility if volatility != 0 else 0

        sortino = calculate_sortino_ratio(portfolio_return)
        max_dd = calculate_max_drawdown(cumulative)
        cagr = (cumulative[-1]) ** (252/len(portfolio_return)) - 1
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        beta = calculate_beta(portfolio_return, sp500_returns)

        # ✅ Výpis metrik
        st.subheader("📊 Portfolio metriky")
        st.write(f"📈 **Očekávaný roční výnos:** {avg_return:.2%}")
        st.write(f"⚠️ **Volatilita:** {volatility:.2%}")
        st.write(f"📊 **Sharpe ratio:** {sharpe:.3f}")
        st.write(f"📉 **Sortino ratio:** {sortino:.3f}")
        st.write(f"📉 **Max Drawdown:** {max_dd:.1%}")
        st.write(f"📈 **CAGR:** {cagr:.2%}")
        st.write(f"📈 **Beta vůči S&P 500:** {beta:.2f}")
        st.write(f"📊 **Calmar ratio:** {calmar:.3f}")

        # ✅ graf vývoje portfolia vs. S&P 500
        st.subheader("📉 Portfolio vs. S&P 500")
        cumulative_sp500 = (1 + sp500_returns).cumprod()

        fig, ax = plt.subplots(figsize=(10,5))
        cumulative.plot(ax=ax, color="blue", label="Portfolio")
        cumulative_sp500.plot(ax=ax, color="orange", label="S&P 500")
        ax.set_title("📉 Portfolio vs. S&P 500")
        ax.set_ylabel("Hodnota (start=1)")
        ax.legend()
        st.pyplot(fig)

        # ✅ pie chart složení portfolia
        st.subheader("🥧 Složení portfolia")
        fig2, ax2 = plt.subplots()
        ax2.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
        ax2.set_title(f"Portfolio složení – strategie {strategy}")
        st.pyplot(fig2)
from scipy.optimize import minimize

def optimize_portfolio(returns, strategy="max_sharpe"):
    """Optimalizuje portfolio podle strategie: max_sharpe nebo min_vol."""
    n_assets = returns.shape[1]

    def portfolio_metrics(weights):
        port_return = np.sum(returns.mean() * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = port_return / port_vol if port_vol != 0 else 0
        return port_return, port_vol, sharpe

    # 📊 Omezení: součet vah = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))

    if strategy == "max_sharpe":
        # 🎯 maximalizujeme Sharpe ratio
        def neg_sharpe(w):
            return -portfolio_metrics(w)[2]
        result = minimize(neg_sharpe, n_assets * [1./n_assets], bounds=bounds, constraints=constraints)
    elif strategy == "min_vol":
        # 🎯 minimalizujeme volatilitu
        def vol(w):
            return portfolio_metrics(w)[1]
        result = minimize(vol, n_assets * [1./n_assets], bounds=bounds, constraints=constraints)
    else:
        return np.array([1./n_assets]*n_assets)  # fallback na equal weight

    return result.x if result.success else np.array([1./n_assets]*n_assets)

def grade_portfolio(sharpe, max_dd, cagr):
    """Vrací známku portfolia (A–F)."""
    score = 0
    # Sharpe ratio (max 40 bodů)
    if sharpe > 1.0:
        score += 40
    elif sharpe > 0.5:
        score += 25
    elif sharpe > 0.2:
        score += 15
    else:
        score += 5

    # Max Drawdown (max 30 bodů)
    if max_dd > -0.1:
        score += 30
    elif max_dd > -0.2:
        score += 20
    else:
        score += 10

    # CAGR (max 30 bodů)
    if cagr > 0.10:
        score += 30
    elif cagr > 0.05:
        score += 20
    else:
        score += 10

    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    elif score >= 25:
        return "E"
    else:
        return "F"

def portfolio_builder_page():
    st.header("💼 Portfolio Builder")

    sp500_df = load_sp500()
    tickers = st.multiselect("📊 Vyber akcie do portfolia:", sp500_df["Symbol"].tolist(), default=["AAPL", "MSFT"])

    strategy = st.selectbox("📈 Vyber strategii:", ["📊 Uživatelské váhy", "⚖️ Equal Weight", "🚀 Max Sharpe", "🛡️ Min Volatility"])

    weights = {}
    if strategy == "📊 Uživatelské váhy":
        st.write("✏️ Zadej váhy jednotlivých akcií (součet = 1.0)")
        for ticker in tickers:
            weights[ticker] = st.number_input(f"Váha pro {ticker}", min_value=0.0, max_value=1.0, value=round(1/len(tickers),2), step=0.05)
    else:
        for ticker in tickers:
            weights[ticker] = 1/len(tickers) if len(tickers) > 0 else 0

    if st.button("📈 Spočítat portfolio"):
        if not tickers:
            st.warning("⚠️ Vyber aspoň jednu akcii.")
            return

        data = yf.download(tickers + ["^GSPC"], period="1y")["Adj Close"]
        sp500 = data["^GSPC"]
        data = data.drop(columns="^GSPC")

        returns = data.pct_change().dropna()
        sp500_returns = sp500.pct_change().dropna()

        # ✅ optimalizace vah pro strategii (pokud není user input)
        if strategy in ["🚀 Max Sharpe", "🛡️ Min Volatility"]:
            opt_strategy = "max_sharpe" if strategy == "🚀 Max Sharpe" else "min_vol"
            optimized_weights = optimize_portfolio(returns, strategy=opt_strategy)
            opt_weights_dict = dict(zip(tickers, optimized_weights))
        else:
            optimized_weights = list(weights.values())
            opt_weights_dict = weights

        # ✅ portfolio return
        portfolio_return = (returns * list(opt_weights_dict.values())).sum(axis=1)
        cumulative = (1 + portfolio_return).cumprod()

        avg_return = portfolio_return.mean() * 252
        volatility = portfolio_return.std() * np.sqrt(252)
        sharpe = avg_return / volatility if volatility != 0 else 0
        sortino = calculate_sortino_ratio(portfolio_return)
        max_dd = calculate_max_drawdown(cumulative)
        cagr = (cumulative[-1]) ** (252/len(portfolio_return)) - 1
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        beta = calculate_beta(portfolio_return, sp500_returns)

        # ✅ známka portfolia
        grade = grade_portfolio(sharpe, max_dd, cagr)

        # 📊 Výpis výsledků
        st.subheader("📊 Hodnocení portfolia")
        st.write(f"📈 **Roční výnos:** {avg_return:.2%}")
        st.write(f"⚠️ **Volatilita:** {volatility:.2%}")
        st.write(f"📊 **Sharpe ratio:** {sharpe:.3f}")
        st.write(f"📉 **Max Drawdown:** {max_dd:.1%}")
        st.write(f"📈 **CAGR:** {cagr:.2%}")
        st.success(f"🎯 **Známka portfolia: {grade}**")

        # 📈 graf portfolia vs. S&P 500
        cumulative_sp500 = (1 + sp500_returns).cumprod()
        fig, ax = plt.subplots(figsize=(10,5))
        cumulative.plot(ax=ax, color="blue", label="Portfolio")
        cumulative_sp500.plot(ax=ax, color="orange", label="S&P 500")
        ax.set_title("📉 Portfolio vs. S&P 500")
        ax.set_ylabel("Hodnota (start=1)")
        ax.legend()
        st.pyplot(fig)

        # 🥧 Pie chart složení
        st.subheader("🥧 Složení portfolia")
        fig2, ax2 = plt.subplots()
        ax2.pie(list(opt_weights_dict.values()), labels=list(opt_weights_dict.keys()), autopct="%1.1f%%")
        ax2.set_title(f"Portfolio složení – strategie {strategy}")
        st.pyplot(fig2)

        # 📊 Tabulka doporučených vah vs. současných
        if strategy == "📊 Uživatelské váhy":
            st.subheader("🎯 Doporučené strategie (optimalizace)")
            max_sharpe_weights = optimize_portfolio(returns, strategy="max_sharpe")
            min_vol_weights = optimize_portfolio(returns, strategy="min_vol")
            st.write("🚀 **Doporučené váhy – Max Sharpe:**")
            st.dataframe(pd.DataFrame({"Ticker": tickers, "Doporučená váha": max_sharpe_weights}))
            st.write("🛡️ **Doporučené váhy – Min Volatility:**")
            st.dataframe(pd.DataFrame({"Ticker": tickers, "Doporučená váha": min_vol_weights}))
                    # 🏆 Slovní komentář k portfoliu
        st.subheader("🏆 Hlubší hodnocení portfolia:")

        # 🔹 Komentář k Sharpe ratio
        if sharpe > 1.0:
            st.write("✅ **Portfolio má velmi dobré Sharpe ratio – výnosy více než kompenzují riziko.**")
        elif sharpe > 0.5:
            st.write("⚖️ **Portfolio má solidní Sharpe ratio – riziko a výnos jsou v rovnováze.**")
        else:
            st.write("❌ **Portfolio má nízké Sharpe ratio – risk neodpovídá výnosům.**")

        # 🔹 Komentář k volatilitě
        if volatility < 0.15:
            st.write("✅ **Portfolio má nízkou volatilitu – stabilní výkonnost.**")
        elif volatility < 0.25:
            st.write("⚖️ **Portfolio má střední volatilitu – přijatelná úroveň rizika.**")
        else:
            st.write("❌ **Portfolio je vysoce volatilní – investice je riziková.**")

        # 🔹 Komentář k Max Drawdown
        if max_dd > -0.1:
            st.write("✅ **Portfolio má nízký maximální propad (Max Drawdown) – dobrá ochrana kapitálu.**")
        elif max_dd > -0.2:
            st.write("⚖️ **Portfolio má střední propad – občasné poklesy, ale zvládnutelné.**")
        else:
            st.write("❌ **Portfolio má velký propad – vysoké riziko ztrát v horších časech.**")

        # 🔹 Komentář k CAGR
        if cagr > 0.10:
            st.write("✅ **Portfolio má vysoký CAGR – velmi dobrý dlouhodobý růst.**")
        elif cagr > 0.05:
            st.write("⚖️ **Portfolio má průměrný CAGR – slušný dlouhodobý výnos.**")
        else:
            st.write("❌ **Portfolio má nízký CAGR – výnos je pod očekáváním.**")

# 📄 valuetracker/screener_ui.py
import streamlit as st

OPERATORS = ["<", "<=", ">", ">=", "==", "!="]

def screener_ui():
    st.header("📊 Stock Screener")

    st.markdown("🎯 **Vyber si ukazatele a filtruj akcie podle svých pravidel!**")

    metrics = [
        "ROA", "ROE", "Debt to Equity", "Current Ratio",
        "Quick Ratio", "Gross Margin"
    ]

    # =======================
    # 📌 SCREENING AKCIÍ BLOK
    # =======================
    st.subheader("📊 Screening akcií")
    criteria = {}

    for i in range(3):
        cols = st.columns([2,1,1])
        with cols[0]:
            metric = st.selectbox(f"Ukazatel {i+1}", ["(žádný)"] + metrics, key=f"metric_{i}")
        with cols[1]:
            operator = st.selectbox("Operátor", OPERATORS, key=f"op_{i}")
        with cols[2]:
            value = st.number_input("Hodnota", value=0.0, key=f"value_{i}")

        if metric != "(žádný)":
            criteria[metric] = (operator, value)

    run_screening = st.button("🔍 Spustit screening")

    # =======================
    # 🏆 RANKING AKCIÍ BLOK
    # =======================
    st.subheader("🏆 Ranking akcií")
    rank_metric = st.selectbox("Rank podle:", metrics, key="rank_metric")

    cols_rank = st.columns([1,2])
    with cols_rank[0]:
        ascending = st.radio("Řazení:", ["Sestupně", "Vzestupně"], index=0) == "Vzestupně"
    with cols_rank[1]:
        top_n = st.slider("Top N:", min_value=1, max_value=50, value=10)

    run_ranking = st.button("🏆 Spustit ranking")

    return criteria, run_screening, rank_metric, ascending, top_n, run_ranking

# 📄 streamlit_app.py

import streamlit as st

# Import hlavních UI funkcí a modulů
from valuetracker.sp500_loader import load_sp500
from valuetracker.movements import get_top_movements_from_csv

# ========== SIDEBAR MENU ==========
st.sidebar.title("📊 Valuetracker – Finance Hub")

menu = st.sidebar.radio(
    "Vyber modul:",
    ["📈 Stock Analysis", "💼 Portfolio Builder", "🔍 Screener", "📊 Top Movements"]
)

# ========== OBSAH STRÁNKY ==========
st.title("📊 Valuetracker")

if menu == "📈 Stock Analysis":
    st.subheader("📈 Analýza vybrané akcie")
    st.info("Vyber akcii ze seznamu a zobrazí se komplexní analýza (TA, fundamentální ukazatele, forecasty, atd.).")
    stock_analysis_page()

elif menu == "💼 Portfolio Builder":
    st.subheader("💼 Tvorba portfolia")
    st.info("Vyber akcie a váhy, nebo použij strategii (Max Sharpe, Min riziko).")

    # 🔄 Načteme seznam akcií z S&P 500
    sp500_df = load_sp500()
    tickers = sp500_df["Symbol"].tolist()

    # 📥 Výběr akcií
    selected_stocks = st.multiselect("Vyber akcie do portfolia:", tickers, default=["AAPL", "MSFT", "GOOGL"])

    # 🏗️ Výběr strategie
    strategy = st.radio("Zvol strategii:", ["Vlastní váhy", "Max Sharpe", "Min Riziko"])

    weights = {}
    if strategy == "Vlastní váhy":
        st.markdown("### ✏️ Nastav váhy jednotlivým akciím")
        total_weight = 0
        for stock in selected_stocks:
            w = st.slider(f"Váha pro {stock} (%)", 0, 100, 10)
            weights[stock] = w / 100
            total_weight += w

        if total_weight != 100:
            st.warning("⚠️ Váhy nedávají dohromady 100 % – portfolio se přepočítá automaticky.")
            # přepočítáme váhy proporcionálně
            total_sum = sum(weights.values())
            weights = {k: v / total_sum for k, v in weights.items()}

    elif strategy == "Max Sharpe":
        st.markdown("📈 **Strategie Max Sharpe zatím nastavuje rovnoměrné váhy. (Budoucí verze přidá optimalizaci)**")
        weights = {stock: 1 / len(selected_stocks) for stock in selected_stocks}

    elif strategy == "Min Riziko":
        st.markdown("🛡️ **Strategie Min Riziko zatím nastavuje rovnoměrné váhy. (Budoucí verze přidá optimalizaci)**")
        weights = {stock: 1 / len(selected_stocks) for stock in selected_stocks}

    # ✅ Tlačítko pro výpočet portfolia
# ✅ Tlačítko pro výpočet portfolia
if st.button("📊 Spočítat portfolio"):
    if not selected_stocks:
        st.error("❌ Vyber alespoň jednu akcii.")
    else:
        st.success("✅ Počítám portfolio…")

        import yfinance as yf

        # 📥 Stáhneme data z Yahoo Finance pro portfolio + S&P 500
        raw_data = yf.download(selected_stocks + ["^GSPC"], period="3y", group_by="ticker", auto_adjust=True)

        # ✅ S&P 500 zvlášť uložíme
        if isinstance(raw_data.columns, pd.MultiIndex):
            sp500 = raw_data["^GSPC"]["Close"]
            data = pd.concat(
                [raw_data[ticker]["Close"].rename(ticker) for ticker in selected_stocks],
                axis=1
            )
        else:
            sp500 = raw_data["Close"]
            data = raw_data[["Close"]].rename(columns={"Close": selected_stocks[0]})

        # ✅ Vyčistíme NaN hodnoty
        data = data.dropna()
        sp500 = sp500.dropna()

        # 🔢 Výpočet denních výnosů
        returns = data.pct_change().dropna()
        sp500_returns = sp500.pct_change().dropna()

        # 📊 Výpočet metrik portfolia
        weights_array = np.array([weights[t] for t in selected_stocks])
        portfolio_returns = (returns * weights_array).sum(axis=1)

        # 📈 Kumulativní vývoj (portfolio i index)
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_sp500 = (1 + sp500_returns).cumprod()

        # 📊 Finanční metriky portfolia
        cagr = (1 + portfolio_returns.mean()) ** 252 - 1
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        volatility = portfolio_returns.std() * np.sqrt(252)
        running_max = cumulative_portfolio.cummax()
        max_dd = (cumulative_portfolio / running_max - 1).min()

        # 📊 Finanční metriky S&P 500 pro srovnání
        sp500_cagr = (1 + sp500_returns.mean()) ** 252 - 1

        # 📈 Graf vývoje portfolia vs. S&P 500
        st.markdown("### 📈 Vývoj portfolia vs. S&P 500")
        fig, ax = plt.subplots(figsize=(10, 5))
        cumulative_portfolio.plot(ax=ax, color="blue", label="Portfolio")
        cumulative_sp500.plot(ax=ax, color="orange", label="S&P 500")
        ax.set_title("Vývoj hodnoty portfolia (3 roky)")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Hodnota (start=1)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # 📊 Tabulka metrik
        st.markdown("### 📊 Hlavní metriky portfolia")
        st.table({
            "CAGR": [f"{cagr:.2%}"],
            "Sharpe ratio": [f"{sharpe:.2f}"],
            "Volatilita": [f"{volatility:.2%}"],
            "Max Drawdown": [f"{max_dd:.2%}"],
            "S&P 500 CAGR": [f"{sp500_cagr:.2%}"]
        })

        # 🏆 Slovní hodnocení
        st.markdown("### 🏆 Hodnocení portfolia")

        # 🔹 Sharpe ratio komentář
        if sharpe > 1.0:
            st.write("✅ **Výborné Sharpe ratio – portfolio má skvělý poměr výnos/riziko.**")
        elif sharpe > 0.5:
            st.write("⚖️ **Solidní Sharpe ratio – portfolio má přijatelný poměr výnos/riziko.**")
        else:
            st.write("❌ **Nízké Sharpe ratio – riziko není dostatečně kompenzováno výnosy.**")

        # 🔹 Volatilita komentář
        if volatility < 0.15:
            st.write("✅ **Nízká volatilita – portfolio je stabilní.**")
        elif volatility < 0.25:
            st.write("⚖️ **Střední volatilita – portfolio má občasné výkyvy, ale není extrémně rizikové.**")
        else:
            st.write("❌ **Vysoká volatilita – portfolio je rizikové.**")

        # 🔹 Max Drawdown komentář
        if max_dd > -0.1:
            st.write("✅ **Nízký maximální propad – portfolio dobře chrání kapitál.**")
        elif max_dd > -0.2:
            st.write("⚖️ **Střední maximální propad – občasné ztráty, ale snesitelné.**")
        else:
            st.write("❌ **Velký maximální propad – portfolio může zaznamenat vysoké ztráty.**")

        # 🔹 CAGR komentář
        if cagr > 0.10:
            st.write("✅ **Vysoký CAGR – portfolio roste velmi dobře dlouhodobě.**")
        elif cagr > 0.05:
            st.write("⚖️ **Průměrný CAGR – portfolio má slušný růst.**")
        else:
            st.write("❌ **Nízký CAGR – portfolio roste pomalu.**")

        # 📊 Porovnání s S&P 500
        st.markdown("### 📊 Jak si portfolio vede proti S&P 500?")
        if cagr > sp500_cagr:
            st.success(f"🚀 **Portfolio překonává S&P 500 o {cagr - sp500_cagr:.2%} ročně!**")
        elif cagr == sp500_cagr:
            st.info("⚖️ **Portfolio má stejný výkon jako S&P 500.**")
        else:
            st.error(f"📉 **Portfolio zaostává za S&P 500 o {sp500_cagr - cagr:.2%} ročně.**")


elif menu == "🔍 Screener":
    st.subheader("🔍 Screener akcií")
    st.info("Filtruj akcie podle kritérií a vytvoř vlastní seznam.")

    # ✅ Načteme seznam S&P 500
    sp500_df = load_sp500()
    tickers = sp500_df["Symbol"].tolist()

    # 📊 Metriky pro screening
    metrics = ["ROA", "ROE", "Debt to Equity", "Current Ratio", "Quick Ratio", "Gross Margin"]

    # 📌 operátory
    OPERATORS = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b
    }

    # =====================================
    # 🎯 Screening akcií – UI
    # =====================================
    st.markdown("### 🎯 Screening akcií")
    st.write("Vyber až 3 kritéria a filtruj akcie:")

    criteria = {}
    for i in range(3):
        cols = st.columns([2, 1, 1])
        with cols[0]:
            metric = st.selectbox(f"Ukazatel {i+1}", ["(žádný)"] + metrics, key=f"metric_{i}")
        with cols[1]:
            operator = st.selectbox("Operátor", list(OPERATORS.keys()), key=f"op_{i}")
        with cols[2]:
            value = st.number_input("Hodnota", value=0.0, key=f"value_{i}")

        if metric != "(žádný)":
            criteria[metric] = (operator, value)

    run_screening = st.button("🔍 Spustit screening")

    st.markdown("---")
    # =====================================
    # 🏆 Ranking akcií – UI
    # =====================================
    st.markdown("### 🏆 Ranking akcií")
    rank_metric = st.selectbox("Rank podle:", metrics, key="rank_metric")

    cols_rank = st.columns([1, 2])
    with cols_rank[0]:
        ascending = st.radio("Řazení:", ["Sestupně", "Vzestupně"], index=0) == "Vzestupně"
    with cols_rank[1]:
        top_n = st.slider("Top N:", min_value=1, max_value=50, value=10)

    run_ranking = st.button("🏆 Spustit ranking")

    # ==================================================
    # 📌 Helper funkce – výpočet ratios pro ticker
    # ==================================================
    def calculate_ratios_for_ticker(ticker):
        """Vrátí dictionary s ratios pro daný ticker – vždy všechny sloupce."""
        try:
            stock_data = get_stock_data(ticker, period="1y")
            info = stock_data['info']
            financials = stock_data['financials']
            balance = stock_data['balance_sheet']

            # ✅ Použijeme poslední sloupce (jako ve Stock Analysis)
            latest_balance = balance.iloc[:, 0] if not balance.empty else pd.Series()
            latest_financials = financials.iloc[:, 0] if not financials.empty else pd.Series()

            # ✅ Poměrové ukazatele (ochrana proti dělení nulou)
            roa = latest_financials.get("Net Income", np.nan) / latest_balance.get("Total Assets", np.nan)
            roe = latest_financials.get("Net Income", np.nan) / (
                latest_balance.get("Total Assets", np.nan) - latest_balance.get("Total Liabilities", 0)
            ) if latest_balance.get("Total Assets", 0) != 0 else np.nan
            current_ratio = latest_balance.get("Current Assets", np.nan) / latest_balance.get("Current Liabilities", np.nan)
            quick_ratio = (latest_balance.get("Current Assets", np.nan) - latest_balance.get("Inventory", 0)) / latest_balance.get("Current Liabilities", np.nan)
            debt_to_equity = latest_balance.get("Total Liabilities", np.nan) / latest_balance.get("Ordinary Shares Number", np.nan)
            gross_margin = latest_financials.get("Gross Profit", np.nan) / latest_financials.get("Total Revenue", np.nan)

            return {
                "Ticker": ticker,
                "Name": info.get("shortName", "N/A"),
                "ROA": roa,
                "ROE": roe,
                "Current Ratio": current_ratio,
                "Quick Ratio": quick_ratio,
                "Debt to Equity": debt_to_equity,
                "Gross Margin": gross_margin
            }
        except Exception as e:
            st.write(f"❌ Chyba při načítání dat pro {ticker}: {e}")
            return {
                "Ticker": ticker,
                "Name": "N/A",
                "ROA": np.nan,
                "ROE": np.nan,
                "Current Ratio": np.nan,
                "Quick Ratio": np.nan,
                "Debt to Equity": np.nan,
                "Gross Margin": np.nan
            }

    # =================================
    # 🔍 SCREENING – po stisknutí tlačítka
    # =================================
    if run_screening:
        st.subheader("📊 Výsledky screeningu")
        st.info("⏳ Načítám data a filtruju…")

        data_list = [calculate_ratios_for_ticker(t) for t in tickers[:30]]  # 🚀 zatím jen 30 tickerů pro rychlost
        df = pd.DataFrame(data_list)

        # 🎯 aplikujeme kritéria
        for metric, (op, value) in criteria.items():
            if metric not in df.columns:
                st.warning(f"⚠️ Sloupec '{metric}' není v datech – přeskočeno.")
                continue

            func = OPERATORS.get(op)
            if func is None:
                st.warning(f"⚠️ Operátor '{op}' neplatný – přeskočeno.")
                continue

            before_rows = len(df)
            df = df[df[metric].notna() & func(df[metric], value)]
            after_rows = len(df)

            st.write(f"📊 **{metric} {op} {value}** → {before_rows} ➡️ {after_rows}")

        # 📊 Výstup
        if df.empty:
            st.error("❌ Žádná akcie nesplňuje zadaná kritéria.")
        else:
            st.dataframe(df)

    # =================================
    # 🏆 RANKING – po stisknutí tlačítka
    # =================================
    if run_ranking:
        st.subheader(f"🏆 Ranking akcií podle: {rank_metric}")
        st.info("⏳ Načítám data a řadím…")

        data_list = [calculate_ratios_for_ticker(t) for t in tickers[:30]]
        df = pd.DataFrame(data_list)

        if rank_metric in df.columns:
            df_ranked = df.sort_values(by=rank_metric, ascending=ascending).head(top_n)
            st.dataframe(df_ranked)
        else:
            st.warning(f"⚠️ Sloupec '{rank_metric}' není dostupný.")

elif menu == "📊 Top Movements":
    st.subheader("📊 Největší pohyby akcií (za včerejšek)")
    st.info("Zobrazí největší růsty a poklesy cen za poslední obchodní den.")

    import yfinance as yf
    from datetime import datetime, timedelta

    # ✅ Načteme tickery z S&P 500
    sp500_df = load_sp500()
    tickers = sp500_df["Symbol"].tolist()

    # ✅ Získáme data za posledních 5 dní (kvůli víkendům/svátkům)
    data = yf.download(tickers, period="5d", interval="1d")["Close"]

    # ✅ Ošetření, kdyby byl jen 1 ticker (převedeme na DataFrame)
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # ✅ Najdeme poslední a předposlední obchodní den
    if len(data) < 2:
        st.error("❌ Nedostatek dat pro výpočet pohybů.")
    else:
        yesterday = data.iloc[-1]
        day_before = data.iloc[-2]

        # ✅ Spočítáme denní procentní změnu
        daily_change = ((yesterday - day_before) / day_before) * 100

        # ✅ Vyčistíme NaN hodnoty
        daily_change = daily_change.dropna()

        # ✅ Najdeme top 5 růstů a poklesů
        top_up = daily_change.sort_values(ascending=False).head(5)
        top_down = daily_change.sort_values(ascending=True).head(5)

        # ✅ Výpis výsledků
        st.markdown("### 📈 Největší růsty (včera)")
        st.dataframe(top_up.to_frame(name="Změna %").style.format({"Změna %": "{:.2f}%"}))

        st.markdown("### 📉 Největší poklesy (včera)")
        st.dataframe(top_down.to_frame(name="Změna %").style.format({"Změna %": "{:.2f}%"}))




