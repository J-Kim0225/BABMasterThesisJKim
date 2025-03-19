import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(beta_file, returns_file, risk_free_file, market_returns_file):
    beta_values_df = pd.read_csv(beta_file, parse_dates=["Date"]).set_index("Date")
    returns_df = pd.read_csv(returns_file, parse_dates=["date"])
    rf_rates_df = pd.read_csv(risk_free_file, parse_dates=["DATE"]).set_index("DATE") / 100 / 12
    market_returns_df = pd.read_csv(market_returns_file, parse_dates=["Date"]).set_index("Date")
    
    return beta_values_df, returns_df, rf_rates_df, market_returns_df

def filter_technology_firms(returns_df):
    # tech_sic_codes = list(range(3570, 3580)) + list(range(3680, 3690)) + [3695] + \
    #                  list(range(7370, 7373)) + [7373, 7375] + \
    #                  list(range(3622, 3623)) + list(range(3661, 3670)) + \
    #                  list(range(3670, 3680)) + list(range(3810, 3813))
    
    # returns_df = returns_df[~returns_df['siccd'].isin(tech_sic_codes)]
    returns_df = returns_df.pivot(index="date", columns="permno", values="ret")
    returns_df.columns = returns_df.columns.astype(str)
    
    return returns_df

def preprocess_data(beta_values_df, returns_df, rf_rates_df, market_returns_df, start_date, end_date):
    beta_values_df = beta_values_df.resample('M').last().loc[start_date:end_date]
    returns_df = returns_df.resample('M').last().loc[start_date:end_date]
    rf_rates_df = rf_rates_df.resample('M').last().loc[start_date:end_date]
    market_returns_df = market_returns_df.resample('M').last().loc[start_date:end_date]
    
    return beta_values_df, returns_df, rf_rates_df, market_returns_df

def calculate_bab_factor(beta_values_df, returns_df):
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
    
    bab_factor = (low_returns_adjusted - high_returns_adjusted)
    return bab_factor, bab_factor.resample('Y').sum()

def plot_bab_factor(bab_factor, title, xlabel, ylabel, width):
    plt.figure(figsize=(12, 6))
    plt.axhline(0, color='black', linewidth=1)
    
    for i in range(1, len(bab_factor)):
        x_values = [bab_factor.index[i - 1], bab_factor.index[i]]
        y_values = [bab_factor.iloc[i - 1], bab_factor.iloc[i]]
        
        if y_values[0] * y_values[1] < 0:
            x_zero = x_values[0] + (x_values[1] - x_values[0]) * (0 - y_values[0]) / (y_values[1] - y_values[0])
            plt.bar([x_values[0], x_zero], [y_values[0], 0], color='blue' if y_values[0] > 0 else 'gray', width=width)
            plt.bar([x_zero, x_values[1]], [0, y_values[1]], color='blue' if y_values[1] > 0 else 'gray', width=width)
        else:
            color = 'blue' if y_values[0] >= 0 else 'gray'
            plt.bar(x_values, y_values, color=color, width=width)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def main():
    path = os.getcwd()
    BETA_FILE = f"{path}/USResults/us_beta_values.csv"
    RETURNS_FILE = f"{path}/US Data/CRSP_monthly_master_thesis_Kim.csv"
    RISK_FREE_FILE = f"{path}/US Data/tbillrate_daily.csv"
    MKT_RETURNS_FILE = f"{path}/US Data/SP500_rets_2003_2024.csv"
    OUTPUT_FILE = f"{path}/USResults/Prop2/bab_factor_us.csv"
    
    start_date, end_date = '2003-01-01', '2023-12-31'
    
    beta_values_df, returns_df, rf_rates_df, market_returns_df = load_data(BETA_FILE, RETURNS_FILE, RISK_FREE_FILE, MKT_RETURNS_FILE)
    returns_df = filter_technology_firms(returns_df)
    beta_values_df, returns_df, rf_rates_df, market_returns_df = preprocess_data(beta_values_df, returns_df, rf_rates_df, market_returns_df, start_date, end_date)
    
    bab_factor, bab_factor_yearly = calculate_bab_factor(beta_values_df, returns_df)
    
    plot_bab_factor(bab_factor, "Monthly BAB Factor Returns (United States)", "Date", "Return (%)", 40)
    plot_bab_factor(bab_factor_yearly, "Yearly BAB Factor Returns (United States)", "Year", "Return (%)", 350)
    
    print(bab_factor_yearly)
    bab_factor.to_csv(OUTPUT_FILE)
    print("BAB Factor data saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
