
import logging, warnings
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Import modulů
from valuetracker.data import get_stock_data
from valuetracker.sp500_loader import load_sp500
from valuetracker.technicals import evaluate_technicals
from valuetracker.valuation_models import evaluate_academic_models
from valuetracker.forecast import (
    forecast_arima,
    forecast_prophet,
    forecast_holt_winters,
    monte_carlo_simulation,
    plot_forecast,
    plot_monte_carlo,
    evaluate_forecast
)
from valuetracker.econometrics import evaluate_econometrics
from valuetracker.risk_models import evaluate_risk
from valuetracker.intrinsic_value import evaluate_intrinsic_value
from valuetracker.portfolio import build_portfolio
from valuetracker.ratios import get_ratios
from valuetracker.compare import compare_to_sector

# 🎨 ANSI BARVY
GREEN = "\033[92m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ✅ Načtení seznamu S&P 500 společností
sp500_df = load_sp500()
print(f"{GREEN}✅ Načten seznam {len(sp500_df)} společností z S&P 500{RESET}")

def grade_from_score(score: float) -> str:
    """Převede numerické skóre (0–100) na písmenovou známku."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    elif score >= 50:
        return "E"
    else:
        return "F"


def stock_analysis_ui():
    """UI modul pro komplexní analýzu vybrané akcie."""
    all_options = [f"{row['Name']} ({row['Symbol']})" for _, row in sp500_df.iterrows()]

    combo = widgets.Combobox(
        placeholder='Začni psát název firmy nebo ticker…',
        options=all_options,
        description='Akcie:',
        ensure_option=True
    )

    period_dropdown = widgets.Dropdown(
        options=[('6 měsíců', '6mo'), ('Poslední rok', '1y'), ('Poslední 3 roky', '3y'),
                 ('Od začátku roku', 'ytd'), ('Celá historie', 'max')],
        value='1y',
        description='Období grafu:'
    )

    forecast_dropdown = widgets.Dropdown(
        options=[('Bez forecastu', 0), ('30 dní', 30), ('90 dní', 90), ('180 dní', 180)],
        value=0,
        description='Forecast:'
    )

    output = widgets.Output()
    stock_data_cache = {}
    state = {"selected_symbol": None, "selected_name": None}

    def plot_graph(symbol, name, selected_period):
        """Vykreslí cenový graf akcie podle zvoleného období."""
        history = stock_data_cache[symbol]['history'].copy()
        history.index = history.index.tz_localize(None)

        if selected_period == '6mo':
            plot_data = history.loc[history.index >= pd.Timestamp.today() - pd.DateOffset(months=6)]
        elif selected_period == '1y':
            plot_data = history.loc[history.index >= pd.Timestamp.today() - pd.DateOffset(years=1)]
        elif selected_period == '3y':
            plot_data = history.loc[history.index >= pd.Timestamp.today() - pd.DateOffset(years=3)]
        elif selected_period == 'ytd':
            current_year = pd.Timestamp.today().year
            plot_data = history[history.index >= f'{current_year}-01-01']
        else:
            plot_data = history

        plt.figure(figsize=(10, 5))
        plt.plot(plot_data.index, plot_data['Close'], label='Closing Price', color='blue')
        plt.title(f'📈 Vývoj ceny akcie {name} ({symbol})')
        plt.xlabel('Datum')
        plt.ylabel('Cena (USD)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def on_submit(change):
        selection = change['new']
        if selection and "(" in selection:
            name = selection.split(" (")[0]
            symbol = selection.split("(")[-1].replace(")", "")

            with output:
                output.clear_output()

                print(f"{BLUE}📥 Stahuji data pro {name} ({symbol})…{RESET}")
                data = get_stock_data(symbol, period="max")
                stock_data_cache[symbol] = data

                info = data['info']
                history = data['history']
                financials = data['financials']
                balance_sheet = data['balance_sheet']
                cashflow_statement = data['cashflow_statement']

                sector = info.get('sector')

                print(f"{GREEN}✅ Data načtena!{RESET}")
                print(f"{BOLD}📊 Název:{RESET} {info.get('shortName', 'N/A')}")
                print(f"{BOLD}Sektor:{RESET} {sector}")
                print(f"{BOLD}Zaměstnanci:{RESET} {info.get('fullTimeEmployees', 'N/A')}\n")

                # ✅ posledních 5 closing prices
                print(f"{BOLD}📊 Posledních 5 closing prices:{RESET}")
                print(history['Close'].tail(5).iloc[::-1])

                # ✅ graf
                plot_graph(symbol, name, period_dropdown.value)

                # ✅ TECHNICKÁ ANALÝZA
                print(f"\n{BOLD}📊 TECHNICKÁ ANALÝZA:{RESET}")
                ta = evaluate_technicals(history)
                for k, v in ta['signals'].items():
                    color = GREEN if "✅" in v or "📈" in v else RED if "❌" in v or "📉" in v else ORANGE
                    print(f"{k}: {color}{v}{RESET}")
                print(f"➡️ TA hodnocení: {ta['overall']}")

                # 📚 AKADEMICKÉ MODELY
                print(f"\n{BOLD}📚 AKADEMICKÉ MODELY:{RESET}")
                academic = evaluate_academic_models(history)
                for model_name in ["Sharpe Ratio", "Beta", "Treynor Ratio", "Jensen's Alpha", "Black–Scholes (demo call)"]:
                    val = academic[model_name]
                    comment = academic["Komentáře"].get(model_name, "")
                    color = GREEN if "✅" in comment else RED if "❌" in comment else ORANGE
                    print(f"{model_name}: {val} {color}{comment}{RESET}")
                print(f"🎯 Závěr: {academic['Hodnocení']}")

                # 📊 EKONOMETRIE
                print(f"\n{BOLD}📊 EKONOMETRICKÁ ANALÝZA:{RESET}")
                econ = evaluate_econometrics(history)
                for section, results in econ.items():
                    if isinstance(results, dict):
                        print(f"{BLUE}🔎 {section}:{RESET}")
                        for k, v in results.items():
                            print(f"   {k}: {v}")
                print(f"📢 Hodnocení: {econ['Hodnocení']}")

                # ⚠️ RIZIKOVÁ ANALÝZA
                print(f"\n{BOLD}⚠️ RIZIKOVÁ ANALÝZA:{RESET}")
                risk = evaluate_risk(history)
                for k, v in risk.items():
                    if k not in ["Komentáře", "Skóre rizika"]:
                        print(f"{k}: {v}")
                for c in risk["Komentáře"]:
                    color = GREEN if "✅" in c else RED if "❌" in c else ORANGE
                    print(f"   💬 {color}{c}{RESET}")
                print(f"📊 Skóre rizika: {risk['Skóre rizika']}/100")

                # 💰 INTRINSIC VALUE
                print(f"\n{BOLD}💰 INTRINSIC VALUE (DCF, DDM):{RESET}")
                intrinsic = evaluate_intrinsic_value(info, financials, history)
                for c in intrinsic["Komentáře"]:
                    print(f"   💬 {c}")
                status_color = GREEN if "PODHODNOCENÁ" in intrinsic['Status'] else RED if "NADHODNOCENÁ" in intrinsic['Status'] else ORANGE
                print(f"➡️ Status: {status_color}{intrinsic['Status']}{RESET}")

                # ✅ ZOBRAZENÍ FINANČNÍCH VÝKAZŮ
                # ✅ FINANČNÍ VÝKAZY – poslední rok + vybrané položky
                pd.set_option('display.max_rows', None)

                latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
                latest_financials = financials.iloc[:, 0] if not financials.empty else pd.Series()
                latest_cashflow = cashflow_statement.iloc[:, 0] if not cashflow_statement.empty else pd.Series()

                # 🏦 Rozvaha
                print(f"\n{BLUE}🏦 Rozvaha – poslední rok:{RESET}")
                if not latest_balance.empty:
                    for item in [
                        "Total Assets",
                        "Total Non Current Assets",
                        "Net PPE",
                        "Machinery Furniture Equipment",
                        "Land And Improvements",
                        "Properties",
                        "Inventory",
                        "Receivables",
                        "Accounts Receivable",
                        "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents"
                    ]:
                        if item in latest_balance.index:
                            print(f"{item}: {latest_balance[item]:,.0f}")
                else:
                    print(f"{RED}❌ Rozvaha nebyla nalezena.{RESET}")

                # 📈 Výkaz zisku a ztrát
                print(f"\n{BLUE}📈 Výkaz zisku a ztrát – poslední rok:{RESET}")
                if not latest_financials.empty:
                    for item in [
                        "EBITDA",
                        "EBIT",
                        "Total Revenue",
                        "Total Expenses",
                        "Gross Profit",
                        "Net Income"
                    ]:
                        if item in latest_financials.index:
                            print(f"{item}: {latest_financials[item]:,.0f}")
                else:
                    print(f"{RED}❌ Výkaz zisku a ztrát nebyl nalezen.{RESET}")

                # 💵 Cash Flow
                print(f"\n{BLUE}💵 Cash Flow – poslední rok:{RESET}")
                if not latest_cashflow.empty:
                    for item in [
                        "Free Cash Flow",
                        "Cash Flow From Continuing Financing Activities",
                        "Cash Flow From Continuing Investing Activities",
                        "Cash Flow From Continuing Operating Activities",
                        "Net Income From Continuing Operations"
                    ]:
                        if item in latest_cashflow.index:
                            print(f"{item}: {latest_cashflow[item]:,.0f}")
                else:
                    print(f"{RED}❌ Cash Flow nebyl nalezen.{RESET}")


                print(f"\n{BOLD}📊 POMĚROVÉ UKAZATELE:{RESET}")
                try:
                    roa = latest_financials["Net Income"] / latest_balance["Total Assets"] if latest_balance.get("Total Assets", 0) != 0 else None
                    roe = latest_financials["Net Income"] / (latest_balance.get("Total Assets", 0) - latest_balance.get("Total Liabilities", 0)) if (latest_balance.get("Total Assets", 0) - latest_balance.get("Total Liabilities", 0)) != 0 else None
                    current_ratio = latest_balance["Current Assets"] / latest_balance["Current Liabilities"] if latest_balance.get("Current Liabilities", 0) != 0 else None
                    quick_ratio = (latest_balance["Current Assets"] - latest_balance["Inventory"]) / latest_balance["Current Liabilities"] if latest_balance.get("Current Liabilities", 0) != 0 else None
                    debt_to_equity = latest_balance.get("Total Liabilities", 0) / latest_balance.get("Ordinary Shares Number", 1)
                    gross_margin = latest_financials["Gross Profit"] / latest_financials["Total Revenue"] if latest_financials.get("Total Revenue", 0) != 0 else None

                    ratios = {
                        "ROA": roa,
                        "ROE": roe,
                        "Current Ratio": current_ratio,
                        "Quick Ratio": quick_ratio,
                        "Debt to Equity": debt_to_equity,
                        "Gross Margin": gross_margin
                    }

                    for k, v in ratios.items():
                        if v is not None:
                            print(f"{k}: {v:.2f}")
                except Exception as e:
                    print(f"{RED}❌ Chyba při výpočtu poměrových ukazatelů: {e}{RESET}")


                # 🔮 FORECAST (pokud vybrán)
                if forecast_dropdown.value > 0:
                    days = forecast_dropdown.value
                    print(f"\n{BOLD}🔮 FORECAST na {days} dní:{RESET}")
                    arima_forecast = forecast_arima(history, periods=days)
                    prophet_forecast = forecast_prophet(history, periods=days)
                    holt_forecast = forecast_holt_winters(history, periods=days)

                    plot_forecast(history, arima_forecast, prophet_forecast, holt_forecast, days=days)

                    mc_simulation = monte_carlo_simulation(history, simulations=200, days=days)
                    plot_monte_carlo(history, mc_simulation, days=days)

                    comment = evaluate_forecast(arima_forecast)
                    print(f"📢 {comment}")

                # 🏆 CELKOVÉ HODNOCENÍ
                print(f"\n{BOLD}🏆 CELKOVÉ HODNOCENÍ AKCIE:{RESET}")
                total_score = 0
                if "bullish" in ta['overall'].lower():
                    total_score += 18
                elif "neutral" in ta['overall'].lower():
                    total_score += 12
                else:
                    total_score += 6

                if "POZITIVNĚ" in academic['Hodnocení']:
                    total_score += 18
                elif "NEUTRÁLNĚ" in academic['Hodnocení']:
                    total_score += 12
                else:
                    total_score += 6

                if "✅" in econ['Hodnocení']:
                    total_score += 15
                else:
                    total_score += 10

                total_score += (risk['Skóre rizika'] * 0.4)

                if "PODHODNOCENÁ" in intrinsic['Status']:
                    total_score += 8
                elif "NADHODNOCENÁ" in intrinsic['Status']:
                    total_score -= 8

                final_score = max(0, round(total_score, 1))
                grade = grade_from_score(final_score)
                grade_color = GREEN if grade in ["A", "B"] else ORANGE if grade in ["C", "D"] else RED

                print(f"✅ Skóre akcie: {BOLD}{final_score}/100{RESET}")
                print(f"🎓 Známka: {grade_color}{BOLD}{grade}{RESET}")

    # ✅ Reakce na výběr akcie
    combo.observe(on_submit, names='value')

    # ✅ Reakce na změnu období
    def on_period_change(change):
        if state["selected_symbol"]:
            with output:
                output.clear_output(wait=True)
                plot_graph(state["selected_symbol"], state["selected_name"], change['new'])

    period_dropdown.observe(on_period_change, names='value')

    display(combo, period_dropdown, forecast_dropdown, output)


# ======================================
# 📊 FUNKCE – PORTFOLIO BUILDER
# ======================================

def portfolio_ui():
    """UI pro tvorbu portfolia s vlastními váhami a strategiemi."""
    all_options = [f"{row['Name']} ({row['Symbol']})" for _, row in sp500_df.iterrows()]

    # ✅ Výběr akcií
    stock_select = widgets.SelectMultiple(
        options=all_options,
        description="📂 Akcie:",
        layout=widgets.Layout(width='50%', height='200px')
    )

    # ✅ Tlačítko pro potvrzení výběru
    confirm_btn = widgets.Button(description="✅ Potvrdit výběr", button_style="info")

    # ✅ Výstupní boxy
    weights_box = widgets.VBox([])
    sum_label = widgets.HTML("<b>Součet vah: 0 %</b>")
    output = widgets.Output()

    # ✅ Dropdown pro strategii
    strategy_dropdown = widgets.Dropdown(
        options=[
            ("📈 Max Sharpe ratio", "max_sharpe"),
            ("🛡️ Min riziko", "min_volatility"),
            ("⚖️ Equal weight", "equal_weight")
        ],
        description="🎯 Strategie:"
    )

    # ✅ Tlačítko pro sestavení portfolia
    build_btn = widgets.Button(description="📊 Sestavit portfolio", button_style="success")

    # ========== FUNKCE ==========
    stock_weight_inputs = {}

    def update_sum_label(change=None):
        """Přepočítá součet vah a aktualizuje label."""
        total_weight = sum([w.value for w in stock_weight_inputs.values()])
        sum_label.value = f"<b>Součet vah: {total_weight} %</b>"
        if total_weight > 100:
            sum_label.value = f"<b style='color:red;'>Součet vah: {total_weight} % (překročeno!)</b>"
        elif total_weight < 100:
            sum_label.value = f"<b style='color:orange;'>Součet vah: {total_weight} %</b>"
        else:
            sum_label.value = f"<b style='color:green;'>Součet vah: {total_weight} % ✔</b>"

    def on_confirm(_):
        """Po výběru akcií vytvoří pole pro zadávání vah."""
        with output:
            output.clear_output()

        selected = stock_select.value
        if not selected:
            with output:
                print(f"{RED}❌ Vyber alespoň jednu akcii.{RESET}")
            return

        # ✅ Pole pro váhy
        stock_weight_inputs.clear()
        weight_widgets = []
        for s in selected:
            symbol = s.split("(")[-1].replace(")", "")
            weight_input = widgets.BoundedFloatText(
                min=0, max=100, step=1,
                description=f"{symbol}:",
                layout=widgets.Layout(width='200px')
            )
            stock_weight_inputs[symbol] = weight_input
            weight_input.observe(update_sum_label, names='value')
            weight_widgets.append(weight_input)

        weights_box.children = weight_widgets + [sum_label]
        update_sum_label()

    def on_build(_):
      with output:
        output.clear_output()
        tickers = list(stock_weight_inputs.keys())
        weights = [w.value for w in stock_weight_inputs.values()]

        total_weight = sum(weights)
        if total_weight != 100:
            display(HTML(f"<span style='color:red; font-weight:bold;'>❌ Součet vah je {total_weight} %. Musí být přesně 100 %.</span>"))
            return

        print(f"{BLUE}📥 Buduji portfolio pro: {tickers}{RESET}")
        result = build_portfolio(tickers, weights=weights)

        # ✅ ✅ METRIKY – full HTML block
        metrics_html = f"""
        <h3><b>🎯 Portfolio (uživatelské váhy)</b></h3>
        <b>📊 Očekávaný roční výnos:</b> {result['expected_return']*100:.2f}%<br>
        <b>⚠️ Volatilita:</b> {result['volatility']*100:.2f}%<br>
        <b>📈 Sharpe ratio:</b> {result['sharpe_ratio']}<br>
        <b>📊 Sortino ratio:</b> {result['sortino_ratio']}<br>
        <b>📉 Max Drawdown:</b> {result['max_drawdown']}<br>
        <b>📈 CAGR:</b> {result['cagr']*100:.2f}%<br>
        <b>β Beta vůči S&P 500:</b> {result['beta']}<br>
        <b>📊 Calmar ratio:</b> {result['calmar_ratio']}
        """
        display(HTML(metrics_html))

        # ✅ ✅ TABULKA VAH
        weights_df = pd.DataFrame(list(result["weights"].items()), columns=["Ticker", "Váha"])
        display(HTML(weights_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
        ).to_html()))

        # ✅ ✅ GRAF – aktuální složení
        plt.figure(figsize=(6, 6))
        plt.pie(weights_df["Váha"], labels=weights_df["Ticker"], autopct="%1.1f%%")
        plt.title("Portfolio složení – uživatelské váhy")
        plt.show()

        # ✅ ✅ GRAF – vývoj portfolia vs. benchmark
        plt.figure(figsize=(10, 5))
        portfolio_curve = result['portfolio_cumulative'] / result['portfolio_cumulative'].iloc[0]
        benchmark_curve = result['benchmark_cumulative'] / result['benchmark_cumulative'].iloc[0]

        portfolio_curve.plot(label="Portfolio", color="blue")
        benchmark_curve.plot(label="S&P 500", color="orange")
        plt.title("📊 Portfolio vs. S&P 500")
        plt.xlabel("Datum")
        plt.ylabel("Hodnota (start=1)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ✅ ✅ HLUBŠÍ HODNOCENÍ
        evaluation = result["evaluation"]
        eval_html = "<h3><b>🏆 Hlubší hodnocení portfolia:</b></h3>"
        for comment in evaluation["comments"]:
            eval_html += f"💬 {comment}<br>"
        grade = evaluation["grade"]
        grade_color = "green" if grade in ["A", "B"] else "orange" if grade in ["C", "D"] else "red"
        eval_html += f"<br><b>🎓 Známka portfolia:</b> <span style='color:{grade_color}; font-size:18px;'>{grade}</span>"
        display(HTML(eval_html))

        # ✅ ✅ STRATEGIE – doporučené váhy
        print(f"\n🎯 <b>Doporučení strategií:</b>")
        strategy = strategy_dropdown.value
        suggested_weights = result["strategies"][strategy]

        df_suggested = pd.DataFrame(list(suggested_weights.items()), columns=["Ticker", "Doporučená váha"])
        df_suggested["Současná váha"] = df_suggested["Ticker"].map(result["weights"])
        df_suggested["Rozdíl"] = df_suggested["Doporučená váha"] - df_suggested["Současná váha"]

        display(HTML(df_suggested.style.set_table_styles(
            [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
        ).to_html()))

        # ✅ ✅ GRAF – doporučené složení
        plt.figure(figsize=(6, 6))
        plt.pie(df_suggested["Doporučená váha"], labels=df_suggested["Ticker"], autopct="%1.1f%%")
        plt.title(f"Portfolio složení – strategie {strategy}")
        plt.show()


    # ✅ Propojení tlačítek
    confirm_btn.on_click(on_confirm)
    build_btn.on_click(on_build)

    # ✅ Zobrazení UI
    display(stock_select, confirm_btn, weights_box, strategy_dropdown, build_btn, output)
