import pandas as pd
import os
import statsmodels.api as sm

path = os.getcwd()
bab_factor = pd.read_csv(f'{path}USResults/Prop2/bab_factor_us.csv', parse_dates=['Date'])
edrate = pd.read_csv(f'{path}US Data/EDRate0321.csv', parse_dates=['Date'])
sofr = pd.read_csv(f'{path}US Data/SOFR.csv', parse_dates=['Date'])
tbill = pd.read_csv(f'{path}US Data/tbillrate_daily.csv', parse_dates=['DATE'])

# Rename columns for consistency
tbill.rename(columns={'DATE': 'Date', 'TB3MS': 'TBillRate'}, inplace=True)
bab_factor.rename(columns={bab_factor.columns[1]: 'r_BAB'}, inplace=True)

# Set index to Date
bab_factor.set_index('Date', inplace=True)
edrate.set_index('Date', inplace=True)
sofr.set_index('Date', inplace=True)
tbill.set_index('Date', inplace=True)

# Convert daily rates to monthly by taking end-of-month values
tbill_monthly = tbill.resample('M').last()
sofr_monthly = sofr.resample('M').last()

# Use EDRate until 2019, then use SOFR
combined_rate = edrate.rename(columns={'Rate': 'TED_Rate'})
combined_rate.update(sofr_monthly.rename(columns={'SOFR': 'TED_Rate'}))
combined_rate = combined_rate.resample('M').last()

# Compute TED Spread (TED_Rate - TBillRate)
ted_spread = combined_rate.join(tbill_monthly, how='inner')
ted_spread['TED_Spread'] = ted_spread['TED_Rate'] - ted_spread['TBillRate']
ted_spread['Delta_TED'] = ted_spread['TED_Spread'].diff()

# Drop NaN values
ted_spread.dropna(inplace=True)

# Merge with BAB factor
data = bab_factor.join(ted_spread[['TED_Spread', 'Delta_TED']], how='inner')
data.dropna(inplace=True)

# Define independent variables
X = data[['TED_Spread', 'Delta_TED']]
X = sm.add_constant(X)  # Add intercept term

# Define dependent variable
y = data['r_BAB']

# Run regression
model = sm.OLS(y, X).fit()

# Print results
print(model.summary())
