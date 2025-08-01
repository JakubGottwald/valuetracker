import pandas as pd
import operator

# 📌 Mapování operátorů na Python funkce
OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

def screen_stocks(csv_path, criteria):
    """
    🔍 Screening akcií na základě zadaných kritérií.

    Parameters:
        csv_path (str): cesta k CSV souboru
        criteria (dict): např. {"PE Ratio": ("<", 15), "Debt/Equity": ("<=", 0.5)}

    Returns:
        pd.DataFrame: gefiltrované akcie, které splňují kritéria
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Nelze načíst CSV: {e}")
        return pd.DataFrame()

    filtered_df = df.copy()

    for metric, (op, value) in criteria.items():
        if metric not in filtered_df.columns:
            print(f"⚠️ Sloupec '{metric}' nebyl nalezen v CSV – přeskočeno.")
            continue
        try:
            func = OPERATORS.get(op)
            if func is None:
                print(f"⚠️ Neznámý operátor '{op}' – přeskočeno.")
                continue

            # aplikujeme filtr
            filtered_df = filtered_df[func(filtered_df[metric], value)]

        except Exception as e:
            print(f"⚠️ Chyba při filtrování podle {metric}: {e}")

    return filtered_df


def rank_stocks(csv_path, metric, ascending=False, top_n=10):
    """
    🏆 Seřadí akcie podle vybraného ukazatele.

    Parameters:
        csv_path (str): cesta k CSV souboru
        metric (str): podle jakého ukazatele řadit
        ascending (bool): True = vzestupně, False = sestupně
        top_n (int): počet výsledků, které chceme vrátit

    Returns:
        pd.DataFrame: seřazené akcie
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Nelze načíst CSV: {e}")
        return pd.DataFrame()

    if metric not in df.columns:
        print(f"❌ Sloupec '{metric}' nebyl nalezen v CSV.")
        return pd.DataFrame()

    try:
        ranked = df.sort_values(by=metric, ascending=ascending).head(top_n)
        return ranked
    except Exception as e:
        print(f"⚠️ Chyba při řazení: {e}")
        return pd.DataFrame()
