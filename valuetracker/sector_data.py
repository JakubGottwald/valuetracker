import yfinance as yf
import pandas as pd
from valuetracker.ratios import get_ratios
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

def fetch_ticker_data(row):
    """
    Pomocná funkce: stáhne data pro jeden ticker.
    """
    ticker = row['Symbol']
    name = row['Name']
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        ratios = get_ratios(info, financials, balance_sheet)
        ratios['Ticker'] = ticker
        ratios['Name'] = name
        ratios['Sector'] = info.get('sector')
        print(f"✅ {ticker} staženo")
        return ratios
    except Exception as e:
        print(f"❌ {ticker} chyba: {e}")
        return None

import yfinance as yf
import pandas as pd
import time
import os

def download_sector_data(sp500_df, csv_path="sector_ratios.csv", batch_size=50, max_retries=5):
    """
    📥 Stáhne / aktualizuje closing ceny pro všechny firmy v S&P 500 po dávkách.
    
    ✅ FUNKCE:
    - Při prvním spuštění stáhne celou historii (period='max')
    - Při dalších spuštěních stáhne jen posledních 5 dní (rychlejší)
    - Používá batch stahování (defaultně 50 tickerů najednou)
    - Exponential backoff při Too Many Requests
    - Ukládání checkpointů po každém batchi → když notebook spadne, data nezmizí
    - Odstraňuje duplicity podle `Ticker` + `Date`
    
    ⚙️ PARAMETRY:
    - sp500_df: DataFrame se sloupci ['Symbol', 'Name']
    - csv_path: cesta k CSV (defaultně 'sector_ratios.csv')
    - batch_size: kolik tickerů stahovat najednou (defaultně 50)
    - max_retries: kolikrát zkusit batch při chybě (defaultně 5)
    """

    # ✅ Zkusíme načíst existující CSV (pokud existuje)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        print(f"✅ Načteno {len(existing_df)} řádků z {csv_path}. Aktualizuji nová data…")
        csv_exists = True
    else:
        print(f"⚠️ Soubor {csv_path} neexistuje – provádím první kompletní stažení všech dat.")
        existing_df = pd.DataFrame()
        csv_exists = False

    tickers = sp500_df['Symbol'].tolist()
    all_data = []

    # ✅ Rozdělíme tickery na dávky
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"\n📦 Batch {i//batch_size + 1}/{len(tickers)//batch_size + 1}: {batch}")

        # 🆕 Když CSV už existuje, stáhneme jen posledních 5 dní
        period = "5d" if csv_exists else "max"

        # 🔁 Retry s exponential backoff
        retries = 0
        while retries < max_retries:
            try:
                df_batch = yf.download(batch, period=period, group_by='ticker', progress=False)
                if df_batch.empty:
                    raise ValueError("Batch nevrátil žádná data.")
                
                # ✅ Zpracujeme výsledky
                for ticker in batch:
                    try:
                        ticker_df = df_batch[ticker].copy()
                        ticker_df = ticker_df.reset_index()
                        ticker_df = ticker_df[['Date', 'Close']]
                        ticker_df['Ticker'] = ticker
                        all_data.append(ticker_df)
                        print(f"✅ {ticker} staženo ({len(ticker_df)} řádků)")
                    except Exception:
                        # ⚠️ Některé tickery vrátí prázdná data
                        print(f"⚠️ Pro {ticker} nebyla nalezena data.")

                # 💾 ✅ Checkpoint – průběžně ukládáme
                if all_data:
                    df_checkpoint = pd.concat([existing_df] + all_data, ignore_index=True)
                    df_checkpoint.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
                    df_checkpoint.to_csv(csv_path, index=False)
                    print(f"💾 Checkpoint uložen ({len(df_checkpoint)} řádků).")

                # 💤 Krátký spánek, aby se snížila zátěž API
                time.sleep(2)
                break  # ✅ batch hotový, vypadneme z retry loopu

            except Exception as e:
                retries += 1
                wait_time = 5 * (2 ** retries)  # exponential backoff: 10, 20, 40, ...
                print(f"❌ Chyba při stahování batch {i//batch_size + 1}: {e}")
                if retries < max_retries:
                    print(f"⏳ Čekám {wait_time}s a zkusím to znovu (pokusu {retries}/{max_retries})…")
                    time.sleep(wait_time)
                else:
                    print(f"🚨 Batch {i//batch_size + 1} přeskočen po {max_retries} pokusech.")
                    break

    if not all_data:
        print("❌ Nebyla získána žádná nová data.")
        return

    # ✅ Spojíme všechny dávky
    df_new = pd.concat(all_data, ignore_index=True)

    # ✅ Spojíme s existujícím CSV a odstraníme duplicity
    if not existing_df.empty:
        combined = pd.concat([existing_df, df_new], ignore_index=True)
        combined.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
    else:
        combined = df_new

    # ✅ Uložíme zpět do CSV
    combined.to_csv(csv_path, index=False)
    print(f"\n📂 Finální uložení hotovo! CSV obsahuje {len(combined)} řádků.")

