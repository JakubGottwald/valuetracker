def get_ratios(info, financials, balance_sheet):
    """
    Vrátí slovník s poměrovými ukazateli firmy.
    """
    ratios = {}

    # P/E
    ratios['P/E'] = info.get('trailingPE')

    # P/B
    ratios['P/B'] = info.get('priceToBook')

    # Debt to Equity
    ratios['Debt/Equity'] = info.get('debtToEquity')

    # Profit margin & ROE
    ratios['Profit Margin'] = info.get('profitMargins')
    ratios['ROE'] = info.get('returnOnEquity')

    # Current ratio
    try:
        current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
        current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
        ratios['Current Ratio'] = current_assets / current_liabilities if current_liabilities else None
    except:
        ratios['Current Ratio'] = None

    # EV/EBITDA
    try:
        market_cap = info.get('marketCap')
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
        cash = balance_sheet.loc['Cash'].iloc[0] if 'Cash' in balance_sheet.index else 0
        ebitda = info.get('ebitda')
        if market_cap and ebitda and ebitda != 0:
            ev = market_cap + total_debt - cash
            ratios['EV/EBITDA'] = ev / ebitda
        else:
            ratios['EV/EBITDA'] = None
    except:
        ratios['EV/EBITDA'] = None

    return ratios
