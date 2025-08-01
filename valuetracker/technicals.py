import pandas as pd
import numpy as np

def calculate_indicators(history: pd.DataFrame):
    """
    Vypočítá TA indikátory: SMA, EMA, MACD, RSI, Bollinger Bands.
    """
    df = history.copy()

    # 📈 SMA a EMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # 🔄 MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 📊 RSI (14 dní)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 🎯 Bollinger Bands (20 dní)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    return df


def evaluate_technicals(history: pd.DataFrame):
    """
    Vyhodnotí TA indikátory a vrátí signály a celkové hodnocení.
    """
    df = calculate_indicators(history)
    latest = df.iloc[-1]
    signals = {}

    # 📌 SMA signál
    if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
        signals['Trend'] = '📈 Silný růstový trend (nad SMA50 i SMA200)'
        trend_score = 2
    elif latest['Close'] > latest['SMA_50']:
        signals['Trend'] = '↗️ Cena nad SMA50 (mírně bullish)'
        trend_score = 1
    else:
        signals['Trend'] = '📉 Cena pod SMA50 (bearish signál)'
        trend_score = -1

    # 📌 RSI
    if latest['RSI'] > 70:
        signals['RSI'] = '🔥 Překoupeno (RSI > 70)'
        rsi_score = -1
    elif latest['RSI'] < 30:
        signals['RSI'] = '🛒 Přeprodáno (RSI < 30)'
        rsi_score = 1
    else:
        signals['RSI'] = '✅ RSI v normálu (30–70)'
        rsi_score = 0

    # 📌 MACD
    if latest['MACD'] > latest['MACD_signal']:
        signals['MACD'] = '📈 MACD nad signální linií (bullish)'
        macd_score = 1
    else:
        signals['MACD'] = '📉 MACD pod signální linií (bearish)'
        macd_score = -1

    # 📌 Bollinger Bands
    if latest['Close'] > latest['BB_Upper']:
        signals['Bollinger'] = '⚠️ Cena nad horním pásmem – možná korekce'
        bb_score = -1
    elif latest['Close'] < latest['BB_Lower']:
        signals['Bollinger'] = '🟢 Cena pod dolním pásmem – možný odraz nahoru'
        bb_score = 1
    else:
        signals['Bollinger'] = '📊 Cena v pásmu (normální volatilita)'
        bb_score = 0

    # 🏆 Celkové hodnocení
    total_score = trend_score + rsi_score + macd_score + bb_score
    if total_score >= 3:
        overall = "✅ TA hodnocení: Silně bullish"
    elif total_score >= 1:
        overall = "🙂 TA hodnocení: Mírně bullish"
    elif total_score == 0:
        overall = "😐 TA hodnocení: Neutrální"
    elif total_score <= -3:
        overall = "❌ TA hodnocení: Silně bearish"
    else:
        overall = "⚠️ TA hodnocení: Mírně bearish"

    return {
        "signals": signals,
        "score": total_score,
        "overall": overall
    }
