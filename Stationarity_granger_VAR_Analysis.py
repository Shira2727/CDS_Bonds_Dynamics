import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import numpy as np
import os

# Load the dataset
filename = 'cdsita.csv'
filepath = os.path.join(os.path.dirname(__file__), filename)

# Read the CSV file with the appropriate separator and decimal
df_cdsita = pd.read_csv(filepath, sep=';', decimal=',')

# Convert and transform 'convspread' column from string to numeric (multiplied by 100)
df_cdsita['convspread'] = pd.to_numeric(df_cdsita['convspread'].str.replace(',', ''), errors='coerce') * 100

# Function to perform the Augmented Dickey-Fuller (ADF) stationarity test
def test_stationarity(series, name):
	"""
	Performs the ADF test on a time series and prints whether the series is stationary.
	
	Parameters:
	series: pd.Series - The time series to test
	name: str - The name of the series for printing results
	"""
	result = adfuller(series.dropna())  # Remove NaN values before testing
	print(f"ADF Test for {name}: Stat={result[0]:.4f}, P-value={result[1]:.4g}")
	print(f"{name} {'is stationary' if result[1] <= 0.05 else 'is NOT stationary'}\n")

# Stationarity test on the original series
test_stationarity(df_cdsita['convspread'], "convspread")
test_stationarity(df_cdsita['bondspread'], "bondspread")

# Create first differences (Δ) for both series
df_cdsita['d_convspread'] = df_cdsita['convspread'].diff()
df_cdsita['d_bondspread'] = df_cdsita['bondspread'].diff()

# Stationarity test on the differenced series
test_stationarity(df_cdsita['d_convspread'], "Δ convspread")
test_stationarity(df_cdsita['d_bondspread'], "Δ bondspread")

# Granger causality test between the two differenced series
df_granger = df_cdsita[['d_convspread', 'd_bondspread']].dropna()

# Perform Granger causality test with up to 5 lags
maxlags = 5
print("\nRunning Granger causality test...")
test_result = grangercausalitytests(df_granger, maxlags, verbose=True)

# Interpretation of Granger causality test results
print("\nInterpretation of Granger causality test results:")
for lag in range(1, maxlags+1):
	print(f"\nLag {lag}:")
	print(f"If p-value < 0.05, we can conclude that one variable Granger-causes the other.")
	p_value = test_result[lag][0]['ssr_chi2test'][1]
	if p_value < 0.05:
		print(f"For lag {lag}, causality is present between the two series.")
	else:
		print(f"For lag {lag}, there is no causality between the two series.")

# Plot to better understand the relationship between the differences
plt.figure(figsize=(10, 6))

# Plot of first differences for both series
plt.subplot(2, 1, 1)
plt.plot(df_cdsita['d_convspread'], label='Δ convspread', color='blue', alpha=0.7)
plt.plot(df_cdsita['d_bondspread'], label='Δ bondspread', color='red', alpha=0.7)
plt.title("First Differences: Δ convspread vs Δ bondspread")
plt.legend(loc='best')

# Rolling correlation between the first differences
plt.subplot(2, 1, 2)
rolling_corr = df_cdsita['d_convspread'].rolling(window=50).corr(df_cdsita['d_bondspread'])
plt.plot(rolling_corr, label='Rolling correlation between Δ convspread and Δ bondspread', color='green')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Rolling correlation between first differences")
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Final considerations:
# - The Granger causality test allows us to check if one variable "causes" the other in the sense of predicting its future values.
# - If the p-value is less than 0.05, we can say there is a causal relationship between the two series.
# - The rolling correlation shows the dynamic relationship over time: if the correlation is high, the two variables move together.

from statsmodels.tsa.api import VAR

# Select stationary columns for VAR
df_var = df_cdsita[['d_convspread', 'd_bondspread']].dropna()

# Fit a VAR model
model = VAR(df_var)
model_fitted = model.fit(3)  # Using lag 3 (you can change this)

# Impulse Response Function (IRF)
irf = model_fitted.irf(10)  # IRF for 10 periods
irf.plot(orth=False)
plt.show()

# Forecast Error Variance Decomposition (FEVD)
fevd = model_fitted.fevd(10)
fevd.plot()
plt.show()

# VECM TEST

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Select the series of interest (differenced or stationary)
df_johansen = df_cdsita[['d_convspread', 'd_bondspread']].dropna()

# Johansen cointegration test
johansen_test = coint_johansen(df_johansen, det_order=0, k_ar_diff=5)

# Johansen test results
print("Eigenvalue Test Statistic:", johansen_test.lr1)
print("Critical values for the test statistics:", johansen_test.cvt)

from statsmodels.tsa.vector_ar.vecm import VECM

# Build the VECM model
vecm = VECM(df_johansen, k_ar_diff=5, coint_rank=1)
vecm_fitted = vecm.fit()

# Show VECM model results
print(vecm_fitted.summary())
