import pandas as pd
from valuetracker.sp500_loader import load_sp500  # 👈 přidáme import

# 🎨 ANSI BARVY
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def get_top_movements_from_csv(csv_path="sector_ratios.csv", top_n=5):
    """
    Načte sector_ratios.csv a najde největších TOP N růstů a poklesů closing price.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"{RED}❌ Nelze načíst CSV: {e}{RESET}")
        return {"top_up": pd.DataFrame(), "top_down": pd.DataFrame()}

    # ✅ Ověříme, že máme potřebné sloupce
    required_cols = {"Ticker", "Date", "Close"}
    if not required_cols.issubset(df.columns):
        print(f"{RED}❌ CSV musí obsahovat sloupce: {required_cols}{RESET}")
        return {"top_up": pd.DataFrame(), "top_down": pd.DataFrame()}

    # ✅ Nahrajeme seznam S&P 500 firem pro mapování ticker → název
    sp500 = load_sp500()
    ticker_name_map = dict(zip(sp500["Symbol"], sp500["Name"]))

    # ✅ Převedeme datum a seřadíme data
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # ✅ Spočítáme denní změnu closing price
    df["Pct_Change"] = df.groupby("Ticker")["Close"].pct_change() * 100
    df = df.dropna(subset=["Pct_Change"])

    # ✅ Doplníme název firmy (pokud v CSV chybí nebo je NaN)
    if "Name" not in df.columns:
        df["Name"] = df["Ticker"].map(ticker_name_map)
    else:
        df["Name"] = df.apply(
            lambda row: row["Name"] if pd.notna(row["Name"]) else ticker_name_map.get(row["Ticker"]),
            axis=1
        )

    # ✅ Vybereme top N růstů a poklesů
    columns_to_show = ["Name", "Ticker", "Date", "Close", "Pct_Change"]
    top_up = df.nlargest(top_n, "Pct_Change")[columns_to_show]
    top_down = df.nsmallest(top_n, "Pct_Change")[columns_to_show]

    # ✅ Barevný výpis
    print(f"{GREEN}\n📈 Největší růsty (TOP {top_n}):{RESET}")
    for _, row in top_up.iterrows():
        print(f"{GREEN}{row['Name']} ({row['Ticker']}){RESET} | {row['Date'].date()} | Close: {row['Close']:.2f} | 🔼 {row['Pct_Change']:.2f}%")

    print(f"{RED}\n📉 Největší poklesy (TOP {top_n}):{RESET}")
    for _, row in top_down.iterrows():
        print(f"{RED}{row['Name']} ({row['Ticker']}){RESET} | {row['Date'].date()} | Close: {row['Close']:.2f} | 🔻 {row['Pct_Change']:.2f}%")

    return {"top_up": top_up, "top_down": top_down}
