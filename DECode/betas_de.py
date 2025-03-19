import pandas as pd
import numpy as np
import os

def load_and_prepare_data(rates_file, cdax_file, returns_file):
    rates_df = pd.read_csv(rates_file)
    cdax_df = pd.read_excel(cdax_file)
    returns_df = pd.read_csv(returns_file, delimiter=';')

    rates_df['Date'] = pd.to_datetime(rates_df['Date'])
    cdax_df['Date'] = pd.to_datetime(cdax_df['Date'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])

    rates_df.set_index('Date', inplace=True)
    cdax_df.set_index('Date', inplace=True)
    returns_df.set_index('Date', inplace=True)

    return rates_df, cdax_df, returns_df

def resample_and_transform_data(rates_df, cdax_df, returns_df):
    monthly_returns_df = returns_df.resample('ME').mean()
    monthly_cdax_df = cdax_df.resample('ME').mean()
    monthly_rates_df = rates_df.resample('ME').mean()

    monthly_returns_df = np.log(1 + monthly_returns_df / 100)
    monthly_cdax_df['Return'] = np.log(1 + monthly_cdax_df['Return'])
    monthly_rates_df['Price'] = monthly_rates_df['Price'] / 100 / 12

    monthly_cdax_df['Excess Return'] = monthly_cdax_df['Return'] - monthly_rates_df['Price']

    # Filter data for the period 01.01.2003 - 31.12.2023 after resampling
    start_date, end_date = '2003-01-01', '2023-12-31'
    monthly_returns_df = monthly_returns_df[(monthly_returns_df.index >= start_date) & (monthly_returns_df.index <= end_date)]
    monthly_cdax_df = monthly_cdax_df[(monthly_cdax_df.index >= start_date) & (monthly_cdax_df.index <= end_date)]
    monthly_rates_df = monthly_rates_df[(monthly_rates_df.index >= start_date) & (monthly_rates_df.index <= end_date)]

    return monthly_returns_df, monthly_cdax_df, monthly_rates_df

def calculate_shrinkage_beta(monthly_returns_df, monthly_cdax_df, monthly_rates_df, shrinkage_factor=0.6):
    beta_df = pd.DataFrame(index=monthly_returns_df.index, columns=monthly_returns_df.columns)

    for stock in monthly_returns_df.columns:
        stock_excess_return = monthly_returns_df[stock] - monthly_rates_df['Price']

        rolling_correlation = (
            stock_excess_return
            .rolling(window=60, min_periods=36)
            .corr(monthly_cdax_df['Excess Return'])
        )
        stock_volatility = stock_excess_return.rolling(window=12, min_periods=12).std()
        market_volatility = monthly_cdax_df['Excess Return'].rolling(window=12, min_periods=12).std()

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

def main(rates_file, cdax_file, returns_file, output_file):
    rates_df, cdax_df, returns_df = load_and_prepare_data(rates_file, cdax_file, returns_file)
    monthly_returns_df, monthly_cdax_df, monthly_rates_df = resample_and_transform_data(rates_df, cdax_df, returns_df)
    beta_df = calculate_shrinkage_beta(monthly_returns_df, monthly_cdax_df, monthly_rates_df)
    save_beta_to_csv(beta_df, output_file)

path = os.getcwd()
main(
    rates_file=f'{path}/German data/Combined_ECB_Rates_and_Germany_3-Month_Yields.csv',
    cdax_file=f'{path}/German data/cdax_returns_06_2024.xlsx',
    returns_file=f'{path}/German data/DE_total_return_01-2024.csv',
    output_file=f'{path}/DEResults/de_beta_values.csv'
)
