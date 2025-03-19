import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def load_data(file, delimiter=',', index_col='DATE'):
    return pd.read_csv(file, delimiter=delimiter, index_col=index_col, parse_dates=True)


def load_and_process_data(beta_file, returns_file, rf_file, cdax_file, ff_file, start_date, end_date, years_to_remove):
    beta_values_df = load_data(beta_file, index_col='Date')
    returns_df = load_data(returns_file, delimiter=';', index_col='Date')
    rf_rates_df = load_data(rf_file, index_col='Date')
    cdax_returns_df = pd.read_excel(cdax_file, index_col='Date', parse_dates=True)
    fama_french_df = load_data(ff_file)

    rf_rates_df = rf_rates_df / 100 / 12

    beta_values_df = beta_values_df.resample('ME').last()
    returns_df = returns_df.resample('ME').last()
    rf_rates_df = rf_rates_df.resample('ME').last()
    cdax_returns_df = cdax_returns_df.resample('ME').last()
    fama_french_df = fama_french_df.resample('ME').last()

    years_to_remove = [int(year) for year in years_to_remove if year]

    beta_values_df = beta_values_df.loc[start_date:end_date]
    returns_df = returns_df.loc[start_date:end_date]
    rf_rates_df = rf_rates_df.loc[start_date:end_date]
    cdax_returns_df = cdax_returns_df.loc[start_date:end_date]
    fama_french_df = fama_french_df.loc[start_date:end_date]

    beta_values_df = beta_values_df[~beta_values_df.index.year.isin(years_to_remove)]
    returns_df = returns_df[~returns_df.index.year.isin(years_to_remove)]
    rf_rates_df = rf_rates_df[~rf_rates_df.index.year.isin(years_to_remove)]
    cdax_returns_df = cdax_returns_df[~cdax_returns_df.index.year.isin(years_to_remove)]
    fama_french_df = fama_french_df[~fama_french_df.index.year.isin(years_to_remove)]

    return beta_values_df, returns_df, rf_rates_df, cdax_returns_df, fama_french_df


def create_beta_sorted_portfolios(beta_df, num_portfolios):
    portfolios = {}
    portfolio_betas = {i + 1: {} for i in range(num_portfolios)}

    for date in beta_df.index:
        sorted_stocks = beta_df.loc[date].dropna().sort_values()
        num_stocks = len(sorted_stocks)
        stocks_per_portfolio = num_stocks // num_portfolios

        for i in range(num_portfolios):
            start = i * stocks_per_portfolio
            end = (i + 1) * stocks_per_portfolio if i < num_portfolios - 1 else num_stocks
            portfolio_stocks = sorted_stocks.index[start:end]
            portfolios.setdefault(i + 1, {}).update({date: portfolio_stocks})
            portfolio_betas[i + 1][date] = sorted_stocks.loc[
                portfolio_stocks].mean() if not portfolio_stocks.empty else np.nan

    portfolio_betas_df = pd.DataFrame.from_dict(portfolio_betas, orient='index').T.sort_index()
    portfolio_betas_df.index.name = 'Date'
    return portfolios, portfolio_betas_df


def calculate_portfolio_returns(returns_df, portfolios):
    portfolio_returns = {}
    for i, portfolio in portfolios.items():
        returns_list = []
        for date, stocks in portfolio.items():
            valid_stocks = [s for s in stocks if s in returns_df.columns]
            if valid_stocks:
                equal_weights = np.ones(len(valid_stocks)) / len(valid_stocks)
                monthly_returns = returns_df.loc[date, valid_stocks].dropna()
                if len(monthly_returns) > 0:
                    portfolio_return = np.dot(equal_weights[:len(monthly_returns)], monthly_returns)
                    returns_list.append(portfolio_return)
                else:
                    returns_list.append(np.nan)
            else:
                returns_list.append(np.nan)
        portfolio_returns[i] = pd.Series(returns_list, index=returns_df.index)

    portfolio_returns_df = pd.DataFrame(portfolio_returns).sort_index()
    return portfolio_returns_df


def compute_sharpe_ratios(portfolio_returns_df, rf_rates_df):
    excess_returns_df = portfolio_returns_df.sub(rf_rates_df['Price'], axis=0)
    annualized_mean_excess_return = excess_returns_df.mean() * 12
    annualized_volatility = excess_returns_df.std() * np.sqrt(12)
    sharpe_ratios = annualized_mean_excess_return / annualized_volatility
    return sharpe_ratios


def plot_sharpe_ratios(sharpe_ratios):
    plt.figure(figsize=(10, 6))
    sharpe_ratios.plot(kind='bar', title='Annualized Sharpe Ratios of 5 Beta-Sorted Portfolios (Germany)')
    plt.xlabel('Portfolio')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    path = os.getcwd()
    BETA_FILE = f"{path}/DEResults/de_beta_values.csv"
    RETURNS_FILE = f"{path}/German data/DE_total_return_01-2024.csv"
    RISK_FREE_FILE = f"{path}/German data/Combined_ECB_Rates_and_Germany_3-Month_Yields.csv"
    CDAX_RETURNS_FILE = f"{path}/German data/cdax_returns_06_2024.xlsx"
    FAMA_FRENCH_FILE = f"{path}/German data/FF_DEU_Values.csv"

    start_date, end_date = '2003-01-01', '2023-12-31'
    years_to_remove = ['']

    beta_values_df, returns_df, rf_rates_df, cdax_returns_df, fama_french_df = load_and_process_data(
        BETA_FILE, RETURNS_FILE, RISK_FREE_FILE, CDAX_RETURNS_FILE, FAMA_FRENCH_FILE,
        start_date, end_date, years_to_remove)

    portfolios, portfolio_betas_df = create_beta_sorted_portfolios(beta_values_df, num_portfolios=5)
    portfolio_returns_df = calculate_portfolio_returns(returns_df, portfolios)
    sharpe_ratios = compute_sharpe_ratios(portfolio_returns_df, rf_rates_df)

    portfolio_betas_df.to_csv(f"{path}/DEResults/portfolio_betas_2020.csv")
    portfolio_returns_df.to_csv(f"{path}/DEResults/portfolio_returns_2020.csv")

    plot_sharpe_ratios(sharpe_ratios)
    print(sharpe_ratios)


if __name__ == "__main__":
    main()
