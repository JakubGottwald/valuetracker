import numpy as np
import pandas as pd

def calculate_volatility(returns):
    """Roční annualizovaná volatilita."""
    return np.std(returns) * np.sqrt(252)

def calculate_var(returns, confidence=0.95):
    """Value at Risk (VaR) – 95% denní ztráta."""
    return np.percentile(returns, (1-confidence)*100)

def calculate_cvar(returns, confidence=0.95):
    """Conditional Value at Risk (CVaR) – průměrná ztráta, pokud VaR padne."""
    var = calculate_var(returns, confidence)
    return returns[returns < var].mean()

def calculate_max_drawdown(prices):
    """Maximální historický drawdown."""
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()

def calculate_kelly_ratio(returns):
    """Kellyho kritérium – kolik reinvestovat (demo)."""
    mean_return = returns.mean()
    var_return = returns.var()
    if var_return == 0:
        return None
    return mean_return / var_return

def evaluate_risk(history):
    """
    Komplexní vyhodnocení rizikovosti akcie s komentáři a skóre.
    """
    close_prices = history['Close']
    returns = close_prices.pct_change().dropna()

    results = {}

    # 📊 Výpočty
    daily_vol = np.std(returns)
    annual_vol = calculate_volatility(returns)
    var_95 = calculate_var(returns)
    cvar_95 = calculate_cvar(returns)
    mdd = calculate_max_drawdown(close_prices)
    kelly = calculate_kelly_ratio(returns)

    results['Denní volatilita'] = round(daily_vol, 3)
    results['Annualizovaná volatilita'] = round(annual_vol, 3)
    results['VaR 95%'] = round(var_95, 3)
    results['CVaR 95%'] = round(cvar_95, 3)
    results['Max Drawdown'] = f"{mdd:.1%}"
    results['Kelly Criterion'] = round(kelly, 3) if kelly else "N/A"

    # 📢 Hodnocení a scoring
    comments = []
    score = 100

    # 📊 Volatilita
    if annual_vol < 0.15:
        comments.append("✅ Nízká roční volatilita.")
    elif annual_vol < 0.30:
        comments.append("⚠️ Střední volatilita.")
        score -= 10
    else:
        comments.append("❌ Vysoká volatilita.")
        score -= 20

    # 📊 VaR & CVaR
    if var_95 > -0.03:
        comments.append("✅ Denní Value at Risk je nízké.")
    elif var_95 > -0.05:
        comments.append("⚠️ Střední riziko (VaR).")
        score -= 5
    else:
        comments.append("❌ Vysoké riziko ztrát (VaR).")
        score -= 15

    # 📊 Max Drawdown
    mdd_val = abs(float(mdd))
    if mdd_val < 0.20:
        comments.append("✅ Historicky malé propady.")
    elif mdd_val < 0.40:
        comments.append("⚠️ Střední drawdown.")
        score -= 10
    else:
        comments.append("❌ Významné historické propady.")
        score -= 20

    # 📊 Kelly Criterion (spíš info)
    if kelly and kelly > 0:
        comments.append("✅ Kellyho kritérium naznačuje pozitivní očekávání.")
    elif kelly and kelly < 0:
        comments.append("⚠️ Kellyho kritérium negativní (pozor).")

    results['Komentáře'] = comments
    results['Skóre rizika'] = max(score, 0)  # nikdy ne záporné

    return results
