import numpy as np

def discounted_cash_flow(fcf, growth_rate, discount_rate, terminal_growth, years=5):
    """
    Jednoduchý DCF model.
    - fcf: poslední známý free cash flow
    - growth_rate: očekávaný růst FCF na dalších pár let
    - discount_rate: požadovaná návratnost (WACC)
    - terminal_growth: dlouhodobý růst po roce 'years'
    """
    cash_flows = []
    for year in range(1, years+1):
        fcf = fcf * (1 + growth_rate)
        discounted = fcf / ((1 + discount_rate) ** year)
        cash_flows.append(discounted)

    # Terminální hodnota (Gordon Growth Model)
    terminal_value = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    discounted_terminal = terminal_value / ((1 + discount_rate) ** years)

    return sum(cash_flows) + discounted_terminal

def dividend_discount(dividend, growth_rate, discount_rate):
    """
    Gordon Dividend Discount Model (DDM) – jen pro firmy vyplácející dividendu.
    """
    if discount_rate <= growth_rate:
        return None
    return dividend * (1 + growth_rate) / (discount_rate - growth_rate)

def evaluate_intrinsic_value(info, financials, history):
    """
    Vyhodnocení intrinsic value.
    Vrací odhad vnitřní hodnoty na akcii a porovnání s tržní cenou.
    """
    price = history['Close'].iloc[-1]
    comments = []

    # Získáme klíčová data z info a financial statements
    fcf = None
    try:
        fcf = financials.loc["Free Cash Flow"].iloc[0]
    except:
        pass

    dividend = info.get("dividendRate", None)

    # Zjednodušené vstupy (do budoucna můžeme rozšířit)
    discount_rate = 0.09  # 9% jako proxy WACC
    growth_rate = 0.03    # 3% roční růst
    terminal_growth = 0.02

    intrinsic_values = {}

    # ✅ DCF (jen pokud máme FCF)
    if fcf and fcf > 0:
        dcf_value = discounted_cash_flow(fcf, growth_rate, discount_rate, terminal_growth)
        shares_outstanding = info.get("sharesOutstanding", None)
        if shares_outstanding:
            intrinsic_per_share = dcf_value / shares_outstanding
            intrinsic_values["DCF"] = intrinsic_per_share
            comments.append(f"DCF model: ${intrinsic_per_share:.2f} na akcii.")
        else:
            comments.append("❌ Nelze spočítat DCF (chybí počet akcií).")
    else:
        comments.append("❌ Nelze spočítat DCF (chybí FCF).")

    # ✅ DDM (jen pokud je dividenda)
    if dividend and dividend > 0:
        ddm_value = dividend_discount(dividend, 0.03, discount_rate)
        if ddm_value:
            intrinsic_values["DDM"] = ddm_value
            comments.append(f"DDM model: ${ddm_value:.2f} na akcii.")
        else:
            comments.append("❌ Nelze spočítat DDM (diskontní sazba ≤ růst).")

    # ✅ Odhad – průměr modelů
    if intrinsic_values:
        avg_value = np.mean(list(intrinsic_values.values()))
        diff = (avg_value - price) / price
        if diff > 0.15:
            status = f"✅ Akcie je PODHODNOCENÁ o {diff:.1%}."
        elif diff < -0.15:
            status = f"❌ Akcie je NADHODNOCENÁ o {abs(diff):.1%}."
        else:
            status = f"⚖️ Akcie je zhruba FÉROVĚ oceněná."
    else:
        avg_value = None
        status = "ℹ️ Nelze vypočítat intrinsic value."

    return {
        "Intrinsic Value Models": intrinsic_values,
        "Average Intrinsic Value": avg_value,
        "Current Price": price,
        "Status": status,
        "Komentáře": comments
    }
