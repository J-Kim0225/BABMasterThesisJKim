import pandas as pd
import numpy as np
import os

def load_and_prepare_data(rates_file, sp500_file, returns_file):
    rates_df = pd.read_csv(rates_file)
    sp500_df = pd.read_csv(sp500_file)
    returns_df = pd.read_csv(returns_file)

    rates_df['DATE'] = pd.to_datetime(rates_df['DATE'], format='%Y-%m-%d')
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'], format='%m-%d-%y')
    returns_df['date'] = pd.to_datetime(returns_df['date'], format='%d%b%Y')

    rates_df= rates_df.set_index('DATE')
    sp500_df = sp500_df.set_index('Date')

    returns_df = returns_df.pivot(index="date", columns="permno", values="ret")
    returns_df.columns = returns_df.columns.astype(str)

    return rates_df, sp500_df, returns_df

def resample_and_transform_data(rates_df, sp500_df, returns_df):
    sp500_df['Return'] = pd.to_numeric(sp500_df['Return'], errors='coerce')

    # Convert all dates to end-of-month
    sp500_df.index = sp500_df.index + pd.offsets.MonthEnd(0)
    rates_df.index = rates_df.index + pd.offsets.MonthEnd(0)
    returns_df.index = returns_df.index + pd.offsets.MonthEnd(0)

    # monthly_returns_df = returns_df.resample('ME').mean()
    monthly_sp500_df = sp500_df.resample('ME').mean()
    monthly_rates_df = rates_df.resample('ME').mean()

    returns_df = returns_df[returns_df.index.is_month_end]

    common_dates = monthly_sp500_df.index.intersection(monthly_rates_df.index).intersection(returns_df.index)
    monthly_sp500_df = monthly_sp500_df.loc[common_dates]
    monthly_rates_df = monthly_rates_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]

    monthly_sp500_df['Return'] = np.log(1 + monthly_sp500_df['Return'])
    monthly_rates_df['TB3MS'] = pd.to_numeric(monthly_rates_df['TB3MS']) / 100 / 12

    monthly_sp500_df['Excess Return'] = monthly_sp500_df['Return'] - monthly_rates_df['TB3MS']

    monthly_returns_df = np.log(1 + returns_df).sub(monthly_rates_df['TB3MS'], axis=0)

    # Filter data for the period 01.01.2003 - 31.12.2023 after resampling
    start_date, end_date = '2003-01-01', '2023-12-31'
    monthly_returns_df = monthly_returns_df[(monthly_returns_df.index >= start_date) & (monthly_returns_df.index <= end_date)]
    monthly_sp500_df = monthly_sp500_df[(monthly_sp500_df.index >= start_date) & (monthly_sp500_df.index <= end_date)]
    monthly_rates_df = monthly_rates_df[(monthly_rates_df.index >= start_date) & (monthly_rates_df.index <= end_date)]

    return monthly_returns_df, monthly_sp500_df, monthly_rates_df

def calculate_shrinkage_beta(monthly_returns_df, monthly_sp500_df, monthly_rates_df, shrinkage_factor=0.6):
    beta_df = pd.DataFrame(index=monthly_returns_df.index, columns=monthly_returns_df.columns)

    for stock in monthly_returns_df.columns:
        stock_excess_return = monthly_returns_df[stock] - monthly_rates_df['TB3MS']

        rolling_correlation = (
            stock_excess_return
            .rolling(window=60, min_periods=36)
            .corr(monthly_sp500_df['Excess Return'])
        )
        stock_volatility = stock_excess_return.rolling(window=12, min_periods=12).std()
        market_volatility = monthly_sp500_df['Excess Return'].rolling(window=12, min_periods=12).std()

        # Compute time-series estimated beta
        ts_beta = rolling_correlation * (stock_volatility / market_volatility)
        beta_df[stock] = ts_beta

    # Compute cross-sectional mean beta
    beta_xs = beta_df.mean(axis=1)  # Mean beta at each time step

    # Apply Vasicek shrinkage adjustment
    shrinkage_beta_df = beta_df.apply(lambda col: shrinkage_factor * col + (1 - shrinkage_factor) * beta_xs, axis=0)

    return shrinkage_beta_df

def save_beta_to_csv(beta_df, output_file):
    beta_df.to_csv(output_file)

def main(rates_file, sp500_file, returns_file, output_file):
    rates_df, sp500_df, returns_df = load_and_prepare_data(rates_file, sp500_file, returns_file)
    monthly_returns_df, monthly_sp500_df, monthly_rates_df = resample_and_transform_data(rates_df, sp500_df, returns_df)
    beta_df = calculate_shrinkage_beta(monthly_returns_df, monthly_sp500_df, monthly_rates_df)
    save_beta_to_csv(beta_df, output_file)

path = os.getcwd()
main(
    rates_file=f'{path}/US Data/tbillrate_daily.csv',
    sp500_file=f'{path}/US Data/SP500_rets_2003_2024.csv',
    returns_file=f'{path}/US Data/CRSP_monthly_master_thesis_Kim.csv',
    output_file=f'{path}/USResults/us_beta_values.csv'
)
