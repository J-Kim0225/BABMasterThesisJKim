import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd()
BETA_FILE = f"{path}/DEResults/de_beta_values.csv"
RETURNS_FILE = f"{path}/German data/DE_total_return_01-2024.csv"
RISK_FREE_FILE = f"{path}/German data/Combined_ECB_Rates_and_Germany_3-Month_Yields.csv"
CDAX_RETURNS_FILE = f"{path}/German data/cdax_returns_06_2024.xlsx"

start_date, end_date = '2003-01-01', '2023-12-31'


def load_data(file, delimiter=',', index_col='Date'):
    """Loads data from CSV or Excel and sets Date as index."""
    if file.endswith('.csv'):
        return pd.read_csv(file, delimiter=delimiter, index_col=index_col, parse_dates=True)
    elif file.endswith('.xlsx'):
        return pd.read_excel(file, index_col=index_col, parse_dates=True)


def preprocess_data(beta_values_df, returns_df, rf_rates_df, cdax_returns_df):
    """Prepares and resamples data to monthly frequency."""
    rf_rates_df = rf_rates_df / 100 / 12

    beta_values_df = beta_values_df.resample('ME').last()
    returns_df = returns_df.resample('ME').last()
    rf_rates_df = rf_rates_df.resample('ME').last()
    cdax_returns_df = cdax_returns_df.resample('ME').last()

    beta_values_df = beta_values_df.loc[start_date:end_date]
    returns_df = returns_df.loc[start_date:end_date]
    rf_rates_df = rf_rates_df.loc[start_date:end_date]
    cdax_returns_df = cdax_returns_df.loc[start_date:end_date]

    return beta_values_df, returns_df, rf_rates_df, cdax_returns_df


def calculate_bab_factor(beta_values_df, returns_df):
    """Calculates BAB factor returns."""
    ranked_df = beta_values_df.rank(axis=1, method='average')
    total_ranks = ranked_df.sum(axis=1)

    low_beta_weights = ranked_df.divide(total_ranks, axis=0)
    max_ranks = ranked_df.max(axis=1).values.reshape(-1, 1)
    high_beta_weights = (max_ranks - ranked_df).divide(total_ranks, axis=0)

    low_portfolio_returns = (low_beta_weights * returns_df).sum(axis=1)
    high_portfolio_returns = (high_beta_weights * returns_df).sum(axis=1)

    low_beta = beta_values_df.mean(axis=1)
    high_beta = beta_values_df.mean(axis=1)

    low_leverage = 1 / low_beta
    high_leverage = 1 / high_beta

    low_returns_adjusted = low_leverage * low_portfolio_returns
    high_returns_adjusted = high_leverage * high_portfolio_returns

    bab_factor = low_returns_adjusted - high_returns_adjusted
    bab_factor_yearly = bab_factor.resample('Y').sum()

    return bab_factor, bab_factor_yearly


def plot_bab_factor(bab_factor, bab_factor_yearly):
    """Plots Monthly and Yearly BAB Factor Returns."""
    plt.figure(figsize=(12, 6))
    plt.axhline(0, color='black', linewidth=1)

    for i in range(1, len(bab_factor)):
        x_values = [bab_factor.index[i - 1], bab_factor.index[i]]
        y_values = [bab_factor.iloc[i - 1], bab_factor.iloc[i]]
        color = 'blue' if y_values[0] >= 0 else 'gray'
        plt.bar(x_values, y_values, color=color, align='center', width=40)

    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.title("Monthly BAB Factor Returns (Germany)")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.axhline(0, color='black', linewidth=1)

    for i in range(1, len(bab_factor_yearly)):
        x_values = [bab_factor_yearly.index[i - 1], bab_factor_yearly.index[i]]
        y_values = [bab_factor_yearly.iloc[i - 1], bab_factor_yearly.iloc[i]]
        color = 'blue' if y_values[0] >= 0 else 'gray'
        plt.bar(x_values, y_values, color=color, align='center', width=350)

    plt.xlabel("Year")
    plt.ylabel("Return (%)")
    plt.title("Yearly BAB Factor Returns (Germany)")
    plt.show()


def main():
    """Main function to execute the analysis."""
    beta_values_df = load_data(BETA_FILE)
    returns_df = load_data(RETURNS_FILE, delimiter=';')
    rf_rates_df = load_data(RISK_FREE_FILE)
    cdax_returns_df = load_data(CDAX_RETURNS_FILE)

    beta_values_df, returns_df, rf_rates_df, cdax_returns_df = preprocess_data(
        beta_values_df, returns_df, rf_rates_df, cdax_returns_df)

    bab_factor, bab_factor_yearly = calculate_bab_factor(beta_values_df, returns_df)
    plot_bab_factor(bab_factor, bab_factor_yearly)

    # bab_factor.to_csv(f'{path}/DEResults/bab_factor_de.csv')
    print(bab_factor_yearly)
    print("BAB Factor data saved successfully.")


if __name__ == "__main__":
    main()