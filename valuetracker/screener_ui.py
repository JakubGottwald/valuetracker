import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from valuetracker.screener import screen_stocks, rank_stocks

def screener_ui(csv_path="sector_ratios.csv"):
    """
    Interaktivní UI pro screening a ranking akcií na základě CSV souboru.

    ✅ Screening: uživatel si vybere až 3 metriky, operátory a hodnoty.
    ✅ Ranking: uživatel si vybere metriku, pořadí (vzestupně/sestupně) a počet akcií (Top N).
    """
    try:
        # ✅ Načteme CSV
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Soubor {csv_path} nebyl nalezen.")
        return
    except Exception as e:
        print(f"❌ Chyba při načítání CSV: {e}")
        return

    # ✅ Získáme seznam metrik – vše kromě identifikačních sloupců
    exclude_cols = ["Ticker", "Date", "Company", "Sector", "Name", "Close"]
    metrics = [c for c in df.columns if c not in exclude_cols]

    # ================================
    # 🎯 Screening – výběr kritérií
    # ================================
    metric_dropdowns = []
    operator_dropdowns = []
    value_texts = []

    for _ in range(3):  # až 3 kritéria
        metric_dropdown = widgets.Dropdown(
            options=["(žádný)"] + metrics,
            description="Ukazatel:"
        )
        operator_dropdown = widgets.Dropdown(
            options=["<", "<=", ">", ">=", "==", "!="],
            description="Operátor:"
        )
        value_text = widgets.FloatText(description="Hodnota:")

        metric_dropdowns.append(metric_dropdown)
        operator_dropdowns.append(operator_dropdown)
        value_texts.append(value_text)

    # ================================
    # 🏆 Ranking – výběr metriky
    # ================================
    rank_metric_dropdown = widgets.Dropdown(
        options=metrics,
        description="Rank podle:"
    )
    rank_order_toggle = widgets.ToggleButtons(
        options=[("Sestupně", False), ("Vzestupně", True)],
        description="Řazení:"
    )
    rank_top_n = widgets.IntSlider(
        value=10, min=1, max=50,
        description="Top N:"
    )

    # ✅ Tlačítka pro akce
    screen_button = widgets.Button(
        description="🔍 Spustit screening",
        button_style="info"
    )
    rank_button = widgets.Button(
        description="🏆 Spustit ranking",
        button_style="success"
    )

    # ✅ Output box pro zobrazení výsledků
    output = widgets.Output()

    # ================================
    # 📊 FUNKCE – Screening kliknutí
    # ================================
    def on_screen_clicked(_):
        with output:
            clear_output()
            # sesbíráme kritéria
            criteria = {}
            for metric_dd, op_dd, val_text in zip(metric_dropdowns, operator_dropdowns, value_texts):
                if metric_dd.value != "(žádný)":
                    criteria[metric_dd.value] = (op_dd.value, val_text.value)

            if not criteria:
                print("⚠️ Nevybral jsi žádná kritéria.")
                return

            # voláme funkci screen_stocks
            filtered = screen_stocks(csv_path, criteria)
            if filtered.empty:
                print("❌ Žádné akcie nesplňují zadaná kritéria.")
            else:
                print(f"✅ Nalezeno {len(filtered)} akcií:")
                display(filtered[["Ticker"] + list(criteria.keys())])

    # ================================
    # 🏆 FUNKCE – Ranking kliknutí
    # ================================
    def on_rank_clicked(_):
        with output:
            clear_output()
            metric = rank_metric_dropdown.value
            ascending = rank_order_toggle.value
            top_n = rank_top_n.value

            ranked = rank_stocks(csv_path, metric=metric, ascending=ascending, top_n=top_n)
            if ranked.empty:
                print("❌ Nepodařilo se provést ranking – zkontroluj CSV nebo metriku.")
            else:
                order_text = "vzestupně" if ascending else "sestupně"
                print(f"🏆 Top {top_n} akcií podle metriky '{metric}' ({order_text}):")
                display(ranked[["Ticker", metric]])

    # ✅ Propojíme tlačítka s funkcemi
    screen_button.on_click(on_screen_clicked)
    rank_button.on_click(on_rank_clicked)

    # ================================
    # 🖥 UI Layout
    # ================================
    display(widgets.HTML("<h3>📊 Screening akcií</h3>"))
    for i in range(3):
        display(widgets.HBox([metric_dropdowns[i], operator_dropdowns[i], value_texts[i]]))
    display(screen_button)

    display(widgets.HTML("<h3>🏆 Ranking akcií</h3>"))
    display(rank_metric_dropdown, rank_order_toggle, rank_top_n, rank_button)

    display(output)
