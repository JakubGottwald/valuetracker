import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# ==========================
# 📥 STAŽENÍ DAT
# ==========================
def download_price_data(tickers, period="1y"):
    """
    Stáhne historická data pro vybrané tickery a vrátí DataFrame s cenami.
    ✅ Preferuje 'Adj Close', pokud není, použije 'Close'.
    """
    data = yf.download(tickers, period=period, progress=False)
    if 'Adj Close' in data.columns:
        price_data = data['Adj Close']
    elif 'Close' in data.columns:
        price_data = data['Close']
    else:
        raise ValueError("❌ Nebyly nalezeny žádné použitelné sloupce cen (Adj Close ani Close).")
    return price_data.fillna(method='ffill').fillna(method='bfill')

# ==========================
# 📊 FINANČNÍ METRIKY
# ==========================
def calculate_max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()

def calculate_cagr(cumulative_returns):
    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0] - 1
    num_years = len(cumulative_returns) / 252
    return (1 + total_return) ** (1/num_years) - 1

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    downside_returns = returns[returns < 0]
    if downside_returns.std() == 0:
        return np.nan
    return (returns.mean() * 252 - risk_free_rate) / (downside_returns.std() * np.sqrt(252))

def calculate_calmar_ratio(cagr, max_drawdown):
    return cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

def calculate_beta_vs_benchmark(portfolio_returns, benchmark_returns):
    cov = np.cov(portfolio_returns, benchmark_returns)
    return cov[0][1] / cov[1][1]

# ==========================
# 📈 ZÁKLADNÍ VÝKON PORTFOLIA
# ==========================
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate=0.02):
    port_return = np.sum(returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
    return port_return, port_volatility, sharpe

# ==========================
# 🎯 STRATEGIE OPTIMALIZACE
# ==========================
def max_sharpe_ratio(returns, cov_matrix):
    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    def neg_sharpe(weights):
        return -portfolio_performance(weights, returns, cov_matrix)[2]

    result = minimize(neg_sharpe,
        num_assets * [1./num_assets,],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints)
    return result.x

def min_volatility(returns, cov_matrix):
    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    def port_vol(weights):
        return portfolio_performance(weights, returns, cov_matrix)[1]

    result = minimize(port_vol,
        num_assets * [1./num_assets,],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints)
    return result.x

# ==========================
# 🆕 HODNOCENÍ PORTFOLIA
# ==========================
def evaluate_portfolio_deep(result):
    """
    Vytvoří slovní hodnocení portfolia a přidělí známku A-F.
    """
    comments = []
    grade = "C"

    # Silné stránky
    if result['sharpe_ratio'] > 1:
        comments.append("✅ Portfolio má velmi dobrý Sharpe ratio – skvělý poměr výnos/riziko.")
    elif result['sharpe_ratio'] > 0.5:
        comments.append("⚠️ Portfolio má průměrný Sharpe ratio – slušný, ale je prostor pro zlepšení.")
    else:
        comments.append("❌ Portfolio má nízký Sharpe ratio – risk neodpovídá výnosům.")

    if result['volatility'] < 0.15:
        comments.append("✅ Portfolio má nízkou volatilitu – stabilní vývoj.")
    elif result['volatility'] < 0.25:
        comments.append("⚠️ Portfolio má střední volatilitu – přijatelná úroveň rizika.")
    else:
        comments.append("❌ Portfolio má vysokou volatilitu – vysoké riziko.")

    # Známka
    score = (result['sharpe_ratio']*40) + (result['cagr']*30) - (result['volatility']*30)
    if score > 30:
        grade = "A"
    elif score > 20:
        grade = "B"
    elif score > 10:
        grade = "C"
    elif score > 0:
        grade = "D"
    else:
        grade = "F"

    return {"comments": comments, "grade": grade}

# ==========================
# 🏗 HLAVNÍ FUNKCE – BUILD PORTFOLIO
# ==========================
def build_portfolio(tickers, weights=None, strategy="none", period="1y", benchmark_ticker="^GSPC"):
    """
    Postaví portfolio a vrátí metriky + doporučené strategie.

    Parameters:
    -----------
    tickers : list - seznam tickerů
    weights : list nebo None - uživatelské váhy (v %), pokud None -> použijí se strategie
    strategy : str - 'none', 'max_sharpe', 'min_volatility', 'equal_weight'
    """
    # ✅ Stažení dat
    price_data = download_price_data(tickers, period)
    benchmark_data = download_price_data(benchmark_ticker, period)

    returns = price_data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna().squeeze()

    cov_matrix = returns.cov()

    # ✅ Nastavení vah
    if weights is not None:
        if sum(weights) != 100:
            raise ValueError(f"❌ Součet vah musí být 100 %, aktuálně: {sum(weights)} %")
        weights = np.array(weights) / 100  # převod na 0–1
    else:
        if strategy == "equal_weight":
            weights = np.array([1 / len(tickers)] * len(tickers))
        elif strategy == "max_sharpe":
            weights = max_sharpe_ratio(returns, cov_matrix)
        elif strategy == "min_volatility":
            weights = min_volatility(returns, cov_matrix)
        else:
            weights = np.array([1 / len(tickers)] * len(tickers))

    # ✅ Výkon portfolia
    port_return, port_vol, sharpe = portfolio_performance(weights, returns, cov_matrix)

    # ✅ Vývoj hodnoty
    portfolio_daily_returns = (returns * weights).sum(axis=1)
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    # ✅ Další metriky
    max_dd = calculate_max_drawdown(portfolio_cumulative)
    cagr = calculate_cagr(portfolio_cumulative)
    sortino = calculate_sortino_ratio(portfolio_daily_returns)
    calmar = calculate_calmar_ratio(cagr, max_dd)
    beta = calculate_beta_vs_benchmark(portfolio_daily_returns, benchmark_returns)

    # ✅ Doporučení strategií
    max_sharpe_w = max_sharpe_ratio(returns, cov_matrix)
    min_vol_w = min_volatility(returns, cov_matrix)
    equal_w = np.array([1 / len(tickers)] * len(tickers))

    # ✅ Hlubší hodnocení
    result = {
        "weights": dict(zip(tickers, np.round(weights, 3))),
        "expected_return": round(port_return, 3),
        "volatility": round(port_vol, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "beta": round(beta, 3),
        "max_drawdown": f"{max_dd:.1%}",
        "cagr": round(cagr, 3),
        "portfolio_cumulative": portfolio_cumulative,
        "benchmark_cumulative": benchmark_cumulative,
        "strategies": {
            "max_sharpe": dict(zip(tickers, np.round(max_sharpe_w, 3))),
            "min_volatility": dict(zip(tickers, np.round(min_vol_w, 3))),
            "equal_weight": dict(zip(tickers, np.round(equal_w, 3)))
        }
    }

    # ✅ Přidáme slovní hodnocení a známku
    result["evaluation"] = evaluate_portfolio_deep(result)
    return result
