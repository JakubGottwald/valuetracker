import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.ardl import ARDL

def run_adf_test(series):
    """
    Augmented Dickey-Fuller Test pro stacionaritu.
    """
    try:
        result = adfuller(series.dropna())
        return {
            "ADF statistika": round(result[0], 3),
            "p-hodnota": round(result[1], 3),
            "Komentář": "✅ Série je stacionární" if result[1] < 0.05 else "❌ Série není stacionární"
        }
    except Exception as e:
        return {"Error": str(e)}

def run_ljung_box(series, lags=10):
    """
    Ljung-Box test na autokorelaci (kontroluje white noise).
    """
    try:
        lb = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
        pval = lb['lb_pvalue'].iloc[0]
        return {
            "Ljung-Box p-hodnota": round(pval, 3),
            "Komentář": "❌ Silná autokorelace" if pval < 0.05 else "✅ Bez výrazné autokorelace"
        }
    except Exception as e:
        return {"Error": str(e)}

def run_arch_test(series):
    """
    ARCH test pro heteroskedasticitu (změny volatility v čase).
    """
    try:
        test_stat, pval, _, _ = het_arch(series.dropna())
        return {
            "ARCH p-hodnota": round(pval, 3),
            "Komentář": "❌ Variance není konstantní" if pval < 0.05 else "✅ Variance je konstantní"
        }
    except Exception as e:
        return {"Error": str(e)}

def run_demo_ardl(series):
    """
    Ukázkový ARDL model (Close_t závisí na předchozích hodnotách).
    Pro seriózní použití by se měla přidat makroekonomická data.
    """
    try:
        df = pd.DataFrame({"y": series})
        df["lag1"] = df["y"].shift(1)
        df["lag2"] = df["y"].shift(2)
        df = df.dropna()

        # Základní ARDL model (lag 2)
        model = ARDL(df["y"], lags=2, exog=df[["lag1", "lag2"]])
        res = model.fit()

        # pokud model nemá rsquared, spočítáme pseudo R²
        if hasattr(res, "rsquared"):
            r2 = round(res.rsquared, 3)
        else:
            residuals = res.resid
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((df["y"] - df["y"].mean()) ** 2)
            r2 = round(1 - ss_res / ss_tot, 3)

        return {
            "ARDL R²": r2,
            "ARDL AIC": round(res.aic, 2),
            "ARDL BIC": round(res.bic, 2)
        }
    except Exception as e:
        return {"Error": str(e)}

def evaluate_econometrics(history):
    """
    Kompletní ekonometrická analýza closing prices.
    Vrací slovník s výsledky testů a celkovým hodnocením.
    """
    close_series = history['Close']

    results = {}
    results["ADF Test"] = run_adf_test(close_series)
    results["Ljung-Box"] = run_ljung_box(close_series)
    results["ARCH Test"] = run_arch_test(close_series)
    results["ARDL Model"] = run_demo_ardl(close_series)

    # 📝 Celkové shrnutí
    summary_parts = []

    # ADF test
    if "p-hodnota" in results["ADF Test"] and results["ADF Test"]["p-hodnota"] < 0.05:
        summary_parts.append("✅ Data jsou stacionární.")
    else:
        summary_parts.append("❌ Data nejsou stacionární.")

    # Ljung-Box test
    if "Ljung-Box p-hodnota" in results["Ljung-Box"] and results["Ljung-Box"]["Ljung-Box p-hodnota"] < 0.05:
        summary_parts.append("⚠️ Detekována autokorelace.")
    else:
        summary_parts.append("✅ Není výrazná autokorelace.")

    # ARCH test
    if "ARCH p-hodnota" in results["ARCH Test"] and results["ARCH Test"]["ARCH p-hodnota"] < 0.05:
        summary_parts.append("⚠️ Přítomná heteroskedasticita (volatilita).")
    else:
        summary_parts.append("✅ Variance stabilní.")

    # ARDL – slovní hodnocení
    if "ARDL R²" in results["ARDL Model"]:
        if results["ARDL Model"]["ARDL R²"] > 0.7:
            summary_parts.append("✅ ARDL model vysvětluje data velmi dobře.")
        elif results["ARDL Model"]["ARDL R²"] > 0.4:
            summary_parts.append("⚠️ ARDL model má střední vypovídací schopnost.")
        else:
            summary_parts.append("❌ ARDL model vysvětluje data slabě.")

    results["Hodnocení"] = " | ".join(summary_parts)
    return results
