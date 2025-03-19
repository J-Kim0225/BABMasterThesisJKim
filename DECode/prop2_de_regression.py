import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def load_data(file, delimiter=',', index_col='DATE'):
    return pd.read_csv(file, delimiter=delimiter, index_col=index_col, parse_dates=True)


def preprocess_data(df, start_date, end_date, years_to_remove):
    df = df.resample('ME').last()
    df = df.loc[start_date:end_date]
    df = df[~df.index.year.isin(years_to_remove)]
    return df


def compute_ex_ante_beta(beta_values_df):
    return beta_values_df.shift(1).mean(axis=1, skipna=True)


def compute_excess_return(portfolio_df, risk_free_df, column_name):
    return portfolio_df[column_name] - (risk_free_df["Price"])


def compute_sharpe_ratio(excess_return):
    annualized_mean = excess_return.mean() * 12
    annualized_volatility = excess_return.std() * np.sqrt(12)
    return annualized_mean / annualized_volatility


def prepare_regression_data(bab_excess_return, cdax_returns_df, rf_rates_df, fama_french_df):
    regression_data = pd.DataFrame({
        "r_P_excess": bab_excess_return,
        "MKT": cdax_returns_df["Return"] - (rf_rates_df["Price"] / 100 / 12)
    })
    for factor in ["SMB", "HML", "UMD"]:
        if factor in fama_french_df.columns:
            regression_data[factor] = fama_french_df[factor]
    return regression_data.dropna()


def run_regression_model(dependent_var, independent_vars, data):
    X = sm.add_constant(data[independent_vars])
    y = data[dependent_var]
    model = sm.OLS(y, X).fit()
    return model


def main():
    path = '/home/kphukan/Documents/J/jthesis/Code'
    files = {
        "beta_values": f"{path}/DEResults/de_beta_values.csv",
        "returns": f"{path}/German data/DE_total_return_01-2024.csv",
        "rf_rates": f"{path}/German data/Combined_ECB_Rates_and_Germany_3-Month_Yields.csv",
        "cdax_returns": f"{path}/German data/cdax_returns_06_2024.xlsx",
        "bab_factor": f"{path}/DEResults/bab_factor_de.csv",
        "fama_french": f"{path}/German data/FF_DEU_Values.csv"
    }

    # Load data
    beta_values_df = load_data(files["beta_values"], index_col='Date')
    returns_df = load_data(files["returns"], delimiter=';', index_col='Date')
    rf_rates_df = load_data(files["rf_rates"], index_col='Date')
    cdax_returns_df = pd.read_excel(files["cdax_returns"], index_col="Date", parse_dates=True)
    bab_factor_df = load_data(files["bab_factor"], index_col='Date')
    fama_french_df = load_data(files["fama_french"])

    rf_rates_df = rf_rates_df / 100 / 12  # Convert risk-free rates

    # Define date range and years to remove
    start_date, end_date = '2003-01-01', '2023-12-31'
    years_to_remove = [2020]

    # Preprocess datasets
    beta_values_df = preprocess_data(beta_values_df, start_date, end_date, years_to_remove)
    returns_df = preprocess_data(returns_df, start_date, end_date, years_to_remove)
    rf_rates_df = preprocess_data(rf_rates_df, start_date, end_date, years_to_remove)
    cdax_returns_df = preprocess_data(cdax_returns_df, start_date, end_date, years_to_remove)
    fama_french_df = preprocess_data(fama_french_df, start_date, end_date, years_to_remove)
    bab_factor_df = preprocess_data(bab_factor_df, start_date, end_date, years_to_remove)

    # Compute key metrics
    ex_ante_bab_beta = compute_ex_ante_beta(beta_values_df)
    bab_excess_return = compute_excess_return(bab_factor_df, rf_rates_df, "BAB Factor")
    bab_sharpe_ratio = compute_sharpe_ratio(bab_excess_return)
    regression_data = prepare_regression_data(bab_excess_return, cdax_returns_df, rf_rates_df, fama_french_df)

    # Run regressions
    capm_model = run_regression_model("r_P_excess", ["MKT"], regression_data)
    fama_french_3_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML"], regression_data)
    carhart_4_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML", "UMD"], regression_data)

    # Extract regression results
    capm_alpha, capm_r2 = capm_model.params["const"], capm_model.rsquared
    capm_alpha_tstat = capm_model.tvalues["const"]

    fama_french_3_alpha, fama_french_3_r2 = fama_french_3_model.params["const"], fama_french_3_model.rsquared
    fama_french_3_alpha_tstat = fama_french_3_model.tvalues["const"]

    carhart_4_alpha, carhart_4_r2 = carhart_4_model.params["const"], carhart_4_model.rsquared
    carhart_4_alpha_tstat = carhart_4_model.tvalues["const"]

    bab_volatility = bab_excess_return.std()

    # Create results table
    bab_stats_table_full = pd.DataFrame({
        "Portfolio": ["BAB"],
        "Excess Return": [bab_excess_return.mean()],
        "CAPM Alpha": [capm_alpha],
        "CAPM Alpha t-stat": [capm_alpha_tstat],
        "CAPM R²": [capm_r2],
        "Three Factor Alpha": [fama_french_3_alpha],
        "Three Factor Alpha t-stat": [fama_french_3_alpha_tstat],
        "Three Factor R²": [fama_french_3_r2],
        "Four Factor Alpha": [carhart_4_alpha],
        "Four Factor Alpha t-stat": [carhart_4_alpha_tstat],
        "Four Factor R²": [carhart_4_r2],
        "Beta (Ex-Ante)": [ex_ante_bab_beta.mean()],
        "Volatility": [bab_volatility],
        "Sharpe Ratio": [bab_sharpe_ratio]
    })

    # Print results
    print(bab_stats_table_full.to_string(index=False))


if __name__ == "__main__":
    main()
