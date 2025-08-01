import pandas as pd

def compare_to_sector(ticker, sector, ratios_dict):
    """
    Porovná ukazatele firmy s průměrem sektoru.
    """
    df = pd.read_csv("sector_ratios.csv")
    sector_df = df[df['Sector'] == sector]

    comparison = {}
    for ratio, value in ratios_dict.items():
        if ratio not in ['Ticker', 'Name', 'Sector']:
            avg = sector_df[ratio].mean()
            if avg is None or avg != avg:
                status = "N/A"
            else:
                if value is None:
                    status = "N/A"
                else:
                    if ratio in ['Profit Margin', 'ROE', 'Current Ratio']:
                        status = "✅ nadprůměr" if value > avg else "❌ podprůměr"
                    else:
                        status = "✅ pod průměrem" if value < avg else "❌ nad průměrem"

            comparison[ratio] = {
                'Hodnota': value,
                'Průměr sektoru': avg,
                'Hodnocení': status
            }
    return pd.DataFrame(comparison).T
