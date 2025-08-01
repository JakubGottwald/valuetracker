import pandas as pd

SP500_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

def load_sp500():
    """
    Načte seznam firem v S&P 500.
    """
    df = pd.read_csv(SP500_URL)
    df = df.rename(columns={"Security": "Name"})
    return df
