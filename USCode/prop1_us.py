import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sharpe_ratios(annual_sharpe_ratios):
    plt.figure(figsize=(12, 6))
    annual_sharpe_ratios.plot(kind='bar')
    plt.title('Annualized Sharpe Ratios for 10 Portfolios (United States)')
    plt.xlabel('Portfolio')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.xticks(ticks=np.arange(len(annual_sharpe_ratios)), labels=np.arange(1, len(annual_sharpe_ratios) + 1), rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def calculate_shrinkage_beta(monthly_sp500_df, crsp_df, shrinkage_factor=0.6):
    beta_df = pd.DataFrame(index=crsp_df.index, columns=crsp_df.columns)
    for stock in crsp_df.columns:
        stock_excess_return = crsp_df[stock]
        market_excess_return = monthly_sp500_df['Excess Return']
        rolling_correlation = stock_excess_return.rolling(window=60, min_periods=36).corr(market_excess_return)
        stock_volatility = stock_excess_return.rolling(window=12, min_periods=12).std()
        market_volatility = market_excess_return.rolling(window=12, min_periods=12).std()
        ts_beta = rolling_correlation * (stock_volatility / market_volatility)
        beta_df[stock] = ts_beta
    beta_xs = beta_df.mean(axis=1)
    return beta_df.apply(lambda col: shrinkage_factor * col + (1 - shrinkage_factor) * beta_xs, axis=0)


def load_and_process_data(path):
    sp500_df = pd.read_csv(f"{path}US Data/SP500_rets_2003_2024.csv")
    tbill_df = pd.read_csv(f"{path}US Data/tbillrate_daily.csv")
    crsp_df = pd.read_csv(f"{path}US Data/CRSP_monthly_master_thesis_Kim.csv")

    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'], format='%m-%d-%y')
    tbill_df['DATE'] = pd.to_datetime(tbill_df['DATE'], format='%Y-%m-%d')
    crsp_df['date'] = pd.to_datetime(crsp_df['date'], format='%d%b%Y')

    sp500_df.set_index('Date', inplace=True)
    tbill_df.set_index('DATE', inplace=True)

    sp500_monthly_df = sp500_df.resample('ME').mean()
    tbill_monthly_df = tbill_df.resample('ME').mean()

    crsp_pivot_df = crsp_df.pivot(index='date', columns='permno', values='ret')

    tbill_monthly_df['TB3MS'] = tbill_monthly_df['TB3MS'] / 100 / 12
    sp500_monthly_df['Excess Return'] = sp500_monthly_df['Return'] - tbill_monthly_df['TB3MS']

    return sp500_monthly_df, tbill_monthly_df, crsp_pivot_df


def filter_data(sp500_monthly_df, tbill_monthly_df, years_to_remove, start_date, end_date):
    sp500_monthly_df = sp500_monthly_df.loc[start_date:end_date]
    tbill_monthly_df = tbill_monthly_df.loc[start_date:end_date]

    years_to_remove = [int(year) for year in years_to_remove if year]
    sp500_monthly_df = sp500_monthly_df[~sp500_monthly_df.index.year.isin(years_to_remove)]
    tbill_monthly_df = tbill_monthly_df[~tbill_monthly_df.index.year.isin(years_to_remove)]

    return sp500_monthly_df, tbill_monthly_df


def form_portfolios(latest_betas):
    sorted_betas = latest_betas.sort_values().dropna()
    bins = min(10, sorted_betas.nunique())
    aligned_bins = pd.qcut(sorted_betas.values, bins, labels=False, duplicates='drop')
    return {i: sorted_betas[aligned_bins == i].index.tolist() for i in range(len(np.unique(aligned_bins)))}


def calculate_portfolio_returns(crsp_df, shrinkage_betas, portfolio_dict):
    portfolio_returns = pd.DataFrame(index=crsp_df.index, columns=portfolio_dict.keys())
    portfolio_betas = pd.DataFrame(index=shrinkage_betas.index, columns=portfolio_dict.keys())
    for i in portfolio_dict:
        portfolio_returns[i] = crsp_df[portfolio_dict[i]].mean(axis=1)
        portfolio_betas[i] = shrinkage_betas[portfolio_dict[i]].mean(axis=1)
    return portfolio_returns, portfolio_betas


def compute_annual_sharpe_ratios(portfolio_returns, tbill_monthly_df):
    excess_returns_df = portfolio_returns.sub(tbill_monthly_df['TB3MS'], axis=0)
    annualized_mean_excess_return = excess_returns_df.mean() * 12
    annualized_volatility = excess_returns_df.std() * np.sqrt(12)
    return annualized_mean_excess_return / annualized_volatility


def save_results(path, years_to_remove, portfolio_returns, portfolio_betas):
    monthly_results = pd.concat([portfolio_returns.add_prefix("Return_"), portfolio_betas.add_prefix("Beta_")], axis=1)
    monthly_results.index.name = "Date"
    monthly_results.to_csv(f"{path}/USResults/Prop1/portfolio_betas_returns_{years_to_remove[0]}.csv")


def main():
    path = os.getcwd()
    years_to_remove = ['2020']
    start_date, end_date = '2003-01-01', '2023-12-31'

    sp500_monthly_df, tbill_monthly_df, crsp_winsorized_df = load_and_process_data(path)
    sp500_monthly_df, tbill_monthly_df = filter_data(sp500_monthly_df, tbill_monthly_df, years_to_remove, start_date,
                                                     end_date)

    shrinkage_betas = calculate_shrinkage_beta(sp500_monthly_df, crsp_winsorized_df)
    shrinkage_betas.fillna(method='ffill', inplace=True)

    latest_betas = shrinkage_betas.iloc[-1]
    portfolio_dict = form_portfolios(latest_betas)

    portfolio_returns, portfolio_betas = calculate_portfolio_returns(crsp_winsorized_df, shrinkage_betas,
                                                                     portfolio_dict)
    annual_sharpe_ratios = compute_annual_sharpe_ratios(portfolio_returns, tbill_monthly_df)

    print(annual_sharpe_ratios)
    plot_sharpe_ratios(annual_sharpe_ratios)
    save_results(path, years_to_remove, portfolio_returns, portfolio_betas)

if __name__ == "__main__":
    main()
