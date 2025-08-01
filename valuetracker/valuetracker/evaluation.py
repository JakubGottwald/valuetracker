def evaluate_stock(comparison_df):
    """
    Spočítá skóre a vrátí slovní hodnocení.
    """
    weights = {
        'P/E': 2,
        'P/B': 1,
        'Debt/Equity': 2,
        'Profit Margin': 2,
        'ROE': 2,
        'Current Ratio': 1,
        'EV/EBITDA': 2
    }

    score = 0
    max_score = sum(weights.values())

    for index, row in comparison_df.iterrows():
        rating = row['Hodnocení']
        if "✅" in str(rating):
            score += weights.get(index, 1)

    if score >= 0.75 * max_score:
        return f"✅ Finální vyhodnocení: Akcie je SPÍŠE PODHODNOCENÁ (score {score}/{max_score})."
    elif score >= 0.5 * max_score:
        return f" Finální vyhodnocení: Akcie odpovídá trhu (score {score}/{max_score})."
    else:
        return f"❌ Finální vyhodnocení: Akcie je SPÍŠE NADHODNOCENÁ (score {score}/{max_score})."
