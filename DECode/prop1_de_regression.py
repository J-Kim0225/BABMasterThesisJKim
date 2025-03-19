import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Define file paths
path = os.getcwd()

FILES = {
    "returns": f"{path}/German data/DE_total_return_01-2024.csv",
    "risk_free": f"{path}/German data/Combined_ECB_Rates_and_Germany_3-Month_Yields.csv",
    "cdax": f"{path}/German data/cdax_returns_06_2024.xlsx",
    "fama_french": f"{path}/German data/FF_DEU_Values.csv",
    "portfolios": f"{path}/DEResults/Prop1/portfolio_returns.csv",
    "betas": f"{path}/DEResults/Prop1/portfolio_betas.csv"
}


# Load CSV and Excel files
def load_data(file, delimiter=',', index_col='DATE'):
    return pd.read_csv(file, delimiter=delimiter, index_col=index_col, parse_dates=True)


def load_excel(file, index_col='Date'):
    return pd.read_excel(file, index_col=index_col, parse_dates=True)


# Load datasets
def load_all_data():
    returns_df = load_data(FILES["returns"], delimiter=';', index_col='Date')
    rf_rates_df = load_data(FILES["risk_free"], index_col='Date') / 100 / 12
    cdax_returns_df = load_excel(FILES["cdax"], index_col="Date")
    fama_french_df = load_data(FILES["fama_french"])
    portfolios_df = load_data(FILES["portfolios"], index_col='Date')
    betas_df = load_data(FILES["betas"], index_col='Date')

    return returns_df, rf_rates_df, cdax_returns_df, fama_french_df, portfolios_df, betas_df


# Resample data to end-of-month frequency and remove specific years
def resample_monthly(years_to_remove, *dfs):
    resampled_dfs = [df.resample('M').last() for df in dfs]
    return [df[~df.index.year.isin(years_to_remove)] for df in resampled_dfs]


# Compute excess returns
def compute_excess_return(portfolio_returns, risk_free_rates):
    return portfolio_returns - (risk_free_rates["Price"] / 100 / 12)


# Run regression model
def run_regression_model(dependent_var, independent_vars, data):
    X = sm.add_constant(data[independent_vars])
    y = data[dependent_var]
    model = sm.OLS(y, X).fit()
    return model


# Perform regressions for each portfolio
def analyze_portfolios(portfolios_df, rf_rates_df, cdax_returns_df, fama_french_df, betas_df):
    results = []
    ex_ante_betas_df = betas_df[[col for col in portfolios_df.columns]]
    betas_df = betas_df.drop(columns=ex_ante_betas_df.columns)

    for portfolio in portfolios_df.columns:
        portfolio_excess_return = compute_excess_return(portfolios_df[portfolio], rf_rates_df)
        sharpe_ratio = (portfolio_excess_return.mean() / portfolio_excess_return.std()) * np.sqrt(12)

        regression_data = pd.DataFrame({
            "r_P_excess": portfolio_excess_return,
            "MKT": compute_excess_return(cdax_returns_df["Return"], rf_rates_df)
        })

        for factor in ["SMB", "HML", "UMD"]:
            if factor in fama_french_df.columns:
                regression_data[factor] = fama_french_df[factor]

        regression_data = regression_data.dropna()

        capm_model = run_regression_model("r_P_excess", ["MKT"], regression_data)
        fama_french_3_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML"], regression_data)
        carhart_4_model = run_regression_model("r_P_excess", ["MKT", "SMB", "HML", "UMD"], regression_data)

        ex_ante_beta = ex_ante_betas_df.mean()

        results.append({
            "Portfolio": portfolio,
            "Excess Return": portfolio_excess_return.mean(),
            "CAPM Alpha": capm_model.params["const"],
            "CAPM Alpha t-stat": capm_model.tvalues["const"],
            "CAPM R²": capm_model.rsquared,
            "Three Factor Alpha": fama_french_3_model.params["const"],
            "Three Factor Alpha t-stat": fama_french_3_model.tvalues["const"],
            "Three Factor R²": fama_french_3_model.rsquared,
            "Four Factor Alpha": carhart_4_model.params["const"],
            "Four Factor Alpha t-stat": carhart_4_model.tvalues["const"],
            "Four Factor R²": carhart_4_model.rsquared,
            "Beta (Ex-Ante)": ex_ante_beta.values,
            "Volatility": portfolio_excess_return.std(),
            "Sharpe Ratio": sharpe_ratio
        })

    return pd.DataFrame(results)


# Save and print results
def save_and_print_results(df, output_path):
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))


# Main execution
def main():
    returns_df, rf_rates_df, cdax_returns_df, fama_french_df, portfolios_df, betas_df = load_all_data()
    years_to_remove = [2020]
    returns_df, rf_rates_df, cdax_returns_df, fama_french_df, portfolios_df, betas_df = resample_monthly(
        years_to_remove, returns_df, rf_rates_df, cdax_returns_df, fama_french_df, portfolios_df, betas_df
    )

    results_df = analyze_portfolios(portfolios_df, rf_rates_df, cdax_returns_df, fama_french_df, betas_df)
    save_and_print_results(results_df, f'{path}/DEResults/Prop1/regression_table_{years_to_remove[0]}.csv')


if __name__ == "__main__":
    main()
