import yfinance as yf

def get_stock_data(ticker, period="1y"):
    """
    Stáhne data o akcii (info, financials, balance sheet, cash flow a historická data).
    ✅ Vrací kompletní set dat pro analýzu.
    """
    stock = yf.Ticker(ticker)

    # Základní info
    info = stock.info

    # Finanční výkazy
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow_statement = stock.cashflow

    # Historie ceny akcie
    history = stock.history(period=period)

    return {
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow_statement": cashflow_statement,
        "history": history
    }
