import pandas as pd
import statsmodels.api as sm
import os

# Load datasets
path = os.getcwd()
bab_factor_de = pd.read_csv(f'{path}/DEResults/bab_factor_de.csv', parse_dates=['Date'])
euribor = pd.read_csv(f'{path}/German data/EURIBOR3m.csv', parse_dates=['Date'])
ecb_rates = pd.read_csv(f'{path}/German data/Combined_ECB_Rates_and_Germany_3-Month_Yields.csv', parse_dates=['Date'])

# Rename columns for consistency
bab_factor_de.rename(columns={'BAB Factor': 'r_BAB'}, inplace=True)
euribor.rename(columns={'Rate': 'EURIBOR_3M'}, inplace=True)
ecb_rates.rename(columns={'Price': 'ECB_Rate'}, inplace=True)

# Set Date as index for all datasets
bab_factor_de.set_index('Date', inplace=True)
euribor.set_index('Date', inplace=True)
ecb_rates.set_index('Date', inplace=True)

# Convert ECB rates to monthly (taking end-of-month values)
ecb_rates_monthly = ecb_rates.resample('M').last()

# Compute TED Spread (EURIBOR 3M - ECB Rate)
ted_spread_de = euribor.join(ecb_rates_monthly, how='inner')
ted_spread_de['TED_Spread'] = ted_spread_de['EURIBOR_3M'] - ted_spread_de['ECB_Rate']
ted_spread_de['Delta_TED'] = ted_spread_de['TED_Spread'].diff()

# Drop NaN values
ted_spread_de.dropna(inplace=True)

# Merge with BAB factor for Germany
data_de = bab_factor_de.join(ted_spread_de[['TED_Spread', 'Delta_TED']], how='inner')
data_de.dropna(inplace=True)

# Define independent variables
X_de = data_de[['TED_Spread', 'Delta_TED']]
X_de = sm.add_constant(X_de)  # Add intercept term

# Define dependent variable
y_de = data_de['r_BAB']

# Run regression
model_de = sm.OLS(y_de, X_de).fit()

# Print results
print(model_de.summary())
