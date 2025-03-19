import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_data(file_path, date_col, date_format=None):
    df = pd.read_csv(file_path)
    try:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    except Exception as e:
        print(f"Error parsing dates in {file_path}: {e}")
    df.set_index(date_col, inplace=True)
    return df

def filter_technology_firms(df):
    tech_sic_codes = list(range(3570, 3580)) + list(range(3680, 3690)) + [3695] + \
                     list(range(7370, 7373)) + [7373, 7375] + \
                     list(range(3622, 3623)) + list(range(3661, 3670)) + \
                     list(range(3670, 3680)) + list(range(3810, 3813))
    
    if 'siccd' in df.columns:
        df = df[~df['siccd'].isin(tech_sic_codes)]
    return df

def preprocess_data(df, resample_freq='M', start_date='2015-01-01', end_date='2018-12-31', years_to_remove=None):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.resample(resample_freq).last()
    df = df.loc[start_date:end_date]
    if years_to_remove:
        df = df.loc[~df.index.year.isin(years_to_remove)]
    return df

def calculate_excess_return(portfolio_df, risk_free_df):
    return portfolio_df.iloc[:, 0] - (risk_free_df.iloc[:, 0] / 100 / 12)

def calculate_ex_ante_beta(stock_betas_df):
    return stock_betas_df.shift(1).mean(axis=1, skipna=True) / 10

def calculate_sharpe_ratio(excess_returns):
    annualized_mean = excess_returns.mean() * 12
    annualized_volatility = excess_returns.std() * np.sqrt(12)
    return annualized_mean / annualized_volatility

def run_regression_model(dependent_var, independent_vars, data):
    X = sm.add_constant(data[independent_vars])
    y = data[dependent_var]
    model = sm.OLS(y, X).fit()
    return model

def main():
    path = os.getcwd()
    years_to_remove = []
    files = {
        "returns": f"{path}/US Data/CRSP_monthly_master_thesis_Kim.csv",
        "risk_free": f"{path}/US Data/tbillrate_daily.csv",
        "sp500": f"{path}/US Data/SP500_rets_2003_2024.csv",
        "bab_factor": f"{path}/USResults/Prop2/bab_factor_us.csv",
        "stock_betas": f"{path}/USResults/us_beta_values.csv",
        "fama_french": f"{path}/US Data/US_ff_Values.csv"
    }
    us_returns_df = load_data(files["returns"], "date", "%d%b%Y")
    us_returns_df = filter_technology_firms(us_returns_df)
    us_rf_rates_df = load_data(files["risk_free"], "DATE", None)
    us_sp500_df = load_data(files["sp500"], "Date", "%m-%d-%y")
    us_bab_factor_df = load_data(files["bab_factor"], "Date")
    us_stock_betas_df = load_data(files["stock_betas"], "Date")
    us_fama_french_df = load_data(files["fama_french"], "DATE")
    dfs = [us_returns_df, us_rf_rates_df, us_sp500_df, us_bab_factor_df, us_stock_betas_df, us_fama_french_df]
    for df in dfs:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
    us_returns_df = preprocess_data(us_returns_df, years_to_remove=years_to_remove)
    us_rf_rates_df = preprocess_data(us_rf_rates_df, years_to_remove=years_to_remove)
    us_sp500_df = preprocess_data(us_sp500_df, years_to_remove=years_to_remove)
    us_bab_factor_df = preprocess_data(us_bab_factor_df, years_to_remove=years_to_remove)
    us_stock_betas_df = preprocess_data(us_stock_betas_df, years_to_remove=years_to_remove)
    us_fama_french_df = preprocess_data(us_fama_french_df, years_to_remove=years_to_remove)
    us_bab_excess_return = calculate_excess_return(us_bab_factor_df, us_rf_rates_df)
    ex_ante_us_bab_beta = calculate_ex_ante_beta(us_stock_betas_df)
    us_bab_sharpe_ratio = calculate_sharpe_ratio(us_bab_excess_return)
    us_regression_data = pd.DataFrame({
        "r_P_excess": us_bab_excess_return,
        "MKT": us_sp500_df.iloc[:, 1] - (us_rf_rates_df.iloc[:, 0] / 100 / 12)
    })
    for factor in ["SMB", "HML", "UMD"]:
        if factor in us_fama_french_df.columns:
            us_regression_data[factor] = us_fama_french_df[factor]
    us_regression_data = us_regression_data.dropna()
    us_capm_model = run_regression_model("r_P_excess", ["MKT"], us_regression_data)
    us_fama_french_3_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML"], us_regression_data)
    us_carhart_4_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML", "UMD"], us_regression_data)
    print(us_capm_model.summary())
    print(us_fama_french_3_model.summary())
    print(us_carhart_4_model.summary())

if __name__ == "__main__":
    main()
