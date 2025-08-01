import numpy as np
import pandas as pd
from scipy.stats import linregress
from math import log, sqrt, exp
from scipy.stats import norm
import yfinance as yf

# --- 📊 Jednotlivé výpočty ---
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_return = returns.mean() - risk_free_rate/252
    volatility = returns.std()
    return excess_return / volatility if volatility != 0 else None

def calculate_treynor_ratio(returns, market_returns, beta, risk_free_rate=0.02):
    avg_return = returns.mean() * 252
    return (avg_return - risk_free_rate) / beta if beta != 0 else None

def calculate_beta(stock_returns, market_returns):
    slope, _, _, _, _ = linregress(market_returns, stock_returns)
    return slope

def calculate_jensens_alpha(stock_returns, market_returns, beta, risk_free_rate=0.02):
    avg_stock = stock_returns.mean() * 252
    avg_market = market_returns.mean() * 252
    expected_return = risk_free_rate + beta * (avg_market - risk_free_rate)
    return avg_stock - expected_return

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    else:
        price = K * exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# --- 🏆 HLAVNÍ FUNKCE ---
def evaluate_academic_models(history):
    """
    Vyhodnotí akcii pomocí akademických modelů (Sharpe, Treynor, Jensen, Black–Scholes)
    a přidá slovní hodnocení + celkové shrnutí.
    """
    results = {}
    commentary = {}
    score = 0

    # ✅ Odstraníme časové pásmo (řeší TypeError)
    history = history.copy()
    history.index = history.index.tz_localize(None)

    # denní výnosy akcie
    stock_returns = history['Close'].pct_change().dropna()

    # ✅ stáhneme benchmark S&P 500
    try:
        benchmark = yf.download("^GSPC", start=history.index.min(), end=history.index.max(), progress=False)
        benchmark.index = benchmark.index.tz_localize(None)
        market_returns = benchmark['Close'].pct_change().dropna()
    except Exception as e:
        print(f"⚠️ Nepodařilo se načíst S&P 500 benchmark: {e}")
        market_returns = None

    # --- 📈 Sharpe Ratio ---
    sharpe = calculate_sharpe_ratio(stock_returns)
    results['Sharpe Ratio'] = round(sharpe, 2) if sharpe else "N/A"
    if sharpe is not None:
        if sharpe > 1:
            commentary['Sharpe Ratio'] = "✅ Výborné risk-adjusted výnosy."
            score += 2
        elif sharpe > 0.3:
            commentary['Sharpe Ratio'] = "⚖️ Přijatelné, ale ne špičkové výnosy."
            score += 1
        else:
            commentary['Sharpe Ratio'] = "❌ Velmi nízké risk-adjusted výnosy."
            score -= 1

    # --- 📈 Beta, Treynor, Jensen ---
    if market_returns is not None and not market_returns.empty:
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        stock_ret = aligned.iloc[:, 0]
        market_ret = aligned.iloc[:, 1]

        beta = calculate_beta(stock_ret, market_ret)
        results['Beta'] = round(beta, 2)
        if beta < 0.8:
            commentary['Beta'] = "🛡️ Defenzivní akcie (méně volatilní než trh)."
            score += 1
        elif beta <= 1.2:
            commentary['Beta'] = "⚖️ Stabilní – pohybuje se jako trh."
        else:
            commentary['Beta'] = "⚠️ Růstová/volatilní akcie – více rizika."
            score -= 1

        treynor = calculate_treynor_ratio(stock_ret, market_ret, beta)
        results['Treynor Ratio'] = round(treynor, 2) if treynor else "N/A"
        if treynor is not None:
            if treynor > 0.5:
                commentary['Treynor Ratio'] = "✅ Dobré výnosy na jednotku tržního rizika."
                score += 1
            elif treynor > 0:
                commentary['Treynor Ratio'] = "⚖️ Lehce pozitivní, ale ne přesvědčivé."
            else:
                commentary['Treynor Ratio'] = "❌ Nízká efektivita vůči tržnímu riziku."
                score -= 1

        alpha = calculate_jensens_alpha(stock_ret, market_ret, beta)
        results["Jensen's Alpha"] = round(alpha, 2) if alpha else "N/A"
        if alpha is not None:
            if alpha > 0:
                commentary["Jensen's Alpha"] = "✅ Akcie překonává očekávání trhu."
                score += 2
            else:
                commentary["Jensen's Alpha"] = "❌ Akcie zaostává za očekáváním."
                score -= 1
    else:
        results['Beta'] = "N/A"
        results['Treynor Ratio'] = "N/A"
        results["Jensen's Alpha"] = "N/A"

    # --- 📈 Black–Scholes ---
    bs_price = black_scholes(
        S=history['Close'].iloc[-1],
        K=history['Close'].iloc[-1]*1.05,
        T=30/365,
        r=0.02,
        sigma=stock_returns.std()*sqrt(252),
        option_type="call"
    )
    results["Black–Scholes (demo call)"] = round(bs_price, 2)
    commentary["Black–Scholes (demo call)"] = "ℹ️ Pouze ilustrace ceny opce, nemá hodnotící význam."

    # --- 🎯 Celkové hodnocení ---
    if score >= 3:
        overall = "✅ Akademické modely hodnotí akcii POZITIVNĚ."
    elif score >= 0:
        overall = "⚖️ Akademické modely hodnotí akcii NEUTRÁLNĚ."
    else:
        overall = "❌ Akademické modely hodnotí akcii spíše NEGATIVNĚ."

    results['Hodnocení'] = overall
    results['Komentáře'] = commentary
    return results
