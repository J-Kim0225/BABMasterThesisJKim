import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_data(file_path, delimiter=',', index_col='DATE'):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, delimiter=delimiter, index_col=index_col, parse_dates=True)

def resample_to_monthly(df):
    """Resample DataFrame to end-of-month frequency."""
    return df.resample('ME').last()

def filter_years(df, years_to_remove):
    """Remove specified years from the DataFrame."""
    if df.index.dtype == 'datetime64[ns]':
        return df[~df.index.year.isin(years_to_remove)]
    else:
        raise ValueError("DataFrame index must be datetime64[ns] to filter by year.")

def calculate_excess_return(portfolio_returns, risk_free_rates):
    """Calculate excess returns for a portfolio."""
    return portfolio_returns - (risk_free_rates["TB3MS"] / 100 / 12)

def calculate_t_statistic(series):
    """Compute t-statistic for a given time series."""
    mean_value = series.mean()
    std_error = series.std() / np.sqrt(len(series))
    return mean_value / std_error

def run_regression_model(dependent_var, independent_vars, data):
    """Run OLS regression and return the fitted model."""
    X = sm.add_constant(data[independent_vars])
    y = data[dependent_var]
    model = sm.OLS(y, X).fit()
    return model

def process_portfolio(portfolio, portfolios_df, ex_ante_betas_df, rf_rates_df, sp500_returns_df, fama_french_df):
    """Process each portfolio: calculate excess return, Sharpe ratio, and run regressions with t-stats."""
    portfolio_excess_return = calculate_excess_return(portfolios_df[portfolio], rf_rates_df)
    sharpe_ratio = (portfolio_excess_return.mean() / portfolio_excess_return.std()) * np.sqrt(12)

    # Calculate t-stat for excess return
    excess_return_tstat = calculate_t_statistic(portfolio_excess_return)

    regression_data = pd.DataFrame({
        "r_P_excess": portfolio_excess_return,
        "MKT": sp500_returns_df["Return"] - (rf_rates_df["TB3MS"] / 100 / 12)
    })

    for factor in ["SMB", "HML", "UMD"]:
        if factor in fama_french_df.columns:
            regression_data[factor] = fama_french_df[factor]

    regression_data = regression_data.dropna()

    capm_model = run_regression_model("r_P_excess", ["MKT"], regression_data)
    fama_french_3_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML"], regression_data)
    carhart_4_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML", "UMD"], regression_data)

    portfolio_id = portfolio.replace("Return_", "")
    aligned_beta = ex_ante_betas_df[f'Beta_{portfolio_id}'].reindex(portfolios_df.index)
    ex_ante_beta = aligned_beta.mean() / 10

    return {
        "Portfolio": portfolio,
        "Excess Return": portfolio_excess_return.mean(),
        "Excess Return t-stat": excess_return_tstat,
        "CAPM Alpha": capm_model.params["const"],
        "CAPM Alpha t-stat": capm_model.tvalues["const"],
        "CAPM R²": capm_model.rsquared,
        "Three Factor Alpha": fama_french_3_model.params["const"],
        "Three Factor Alpha t-stat": fama_french_3_model.tvalues["const"],
        "Three Factor R²": fama_french_3_model.rsquared,
        "Four Factor Alpha": carhart_4_model.params["const"],
        "Four Factor Alpha t-stat": carhart_4_model.tvalues["const"],
        "Four Factor R²": carhart_4_model.rsquared,
        "Beta (Ex-Ante)": ex_ante_beta,
        "Volatility": portfolio_excess_return.std(),
        "Sharpe Ratio": sharpe_ratio
    }

def main():
    path = os.getcwd()

    # File paths
    portfolios_file = f'{path}/USResults/Prop1/portfolio_betas_returns.csv'
    fama_french_file = f'{path}/US Data/US_ff_Values.csv'
    returns_file = f"{path}/US Data/CRSP_monthly_master_thesis_Kim.csv"
    risk_free_file = f"{path}/US Data/tbillrate_daily.csv"
    sp500_returns_file = f"{path}/US Data/SP500_rets_2003_2024.csv"

    # Load data
    portfolios_df = load_data(portfolios_file, index_col='Date')
    fama_french_df = load_data(fama_french_file)
    returns_df = load_data(returns_file, index_col='date')
    rf_rates_df = load_data(risk_free_file)
    sp500_returns_df = load_data(sp500_returns_file, index_col='Date')

    # Resample data to monthly frequency
    returns_df = resample_to_monthly(returns_df)
    rf_rates_df = resample_to_monthly(rf_rates_df)
    sp500_returns_df = resample_to_monthly(sp500_returns_df)
    fama_french_df = resample_to_monthly(fama_french_df)
    portfolios_df = resample_to_monthly(portfolios_df)

    # Define years to remove
    years_to_remove = [2020]

    # Filter out specified years
    portfolios_df = filter_years(portfolios_df, years_to_remove)
    fama_french_df = filter_years(fama_french_df, years_to_remove)
    returns_df = filter_years(returns_df, years_to_remove)
    rf_rates_df = filter_years(rf_rates_df, years_to_remove)
    sp500_returns_df = filter_years(sp500_returns_df, years_to_remove)

    # Extract ex-ante betas
    ex_ante_betas_df = portfolios_df[[col for col in portfolios_df.columns if 'Beta_' in col]]
    portfolios_df = portfolios_df.drop(columns=ex_ante_betas_df.columns)

    # Process each portfolio
    results = []
    for portfolio in portfolios_df.columns:
        result = process_portfolio(portfolio, portfolios_df, ex_ante_betas_df, rf_rates_df, sp500_returns_df, fama_french_df)
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print results
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv(f'{path}/USResults/Prop1/regression_table_{years_to_remove[0]}.csv', index=False)

if __name__ == "__main__":
    main()
