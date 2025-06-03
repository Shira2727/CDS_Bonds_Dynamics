import pandas as pd
import matplotlib.pyplot as plt
import os

# Use a relative path to read the CSV file from the same directory as the script
filename = 'cdsita.csv'
filepath = os.path.join(os.path.dirname(__file__), filename)

# Read the CSV file with the appropriate separator and decimal
df_cdsita = pd.read_csv(filepath, sep=';', decimal=',')
print(df_cdsita)

# Parse the 'Date' column as datetime
df_cdsita['Date'] = pd.to_datetime(df_cdsita['Date'], format='%d/%m/%Y')

# Set 'Date' as the index
df_cdsita.set_index('Date', inplace=True)

# Convert 'convspread' to numeric, removing commas and handling errors
df_cdsita['convspread'] = pd.to_numeric(df_cdsita['convspread'].astype(str).str.replace(',', ''), errors='coerce')

# Multiply 'convspread' by 100
df_cdsita['convspread'] = df_cdsita['convspread'] * 100

# Create the 'basis' column
df_cdsita['basis'] = df_cdsita['convspread'] - df_cdsita['bondspread']

# Calculate correlation between convspread and bondspread
correlation = df_cdsita['convspread'].corr(df_cdsita['bondspread'])
print(f"Correlation between convspread and bondspread: {correlation}")

# Calculate standard deviations
convspread_std = df_cdsita['convspread'].std()
bondspread_std = df_cdsita['bondspread'].std()
print(f"Standard deviation of convspread: {convspread_std}")
print(f"Standard deviation of bondspread: {bondspread_std}")

# Plot convspread and bondspread over time
plt.figure(figsize=(10, 5))
plt.plot(df_cdsita['convspread'], label='convspread')
plt.plot(df_cdsita['bondspread'], label='bondspread')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.title('Convspread and Bondspread Over Time')
plt.legend()
plt.show()

# Calculate mean and median
print(f"Mean of convspread: {df_cdsita['convspread'].mean()}")
print(f"Mean of bondspread: {df_cdsita['bondspread'].mean()}")
print(f"Median of convspread: {df_cdsita['convspread'].median()}")
print(f"Median of bondspread: {df_cdsita['bondspread'].median()}")

# Calculate range
print(f"Range of convspread: {df_cdsita['convspread'].max() - df_cdsita['convspread'].min()}")
print(f"Range of bondspread: {df_cdsita['bondspread'].max() - df_cdsita['bondspread'].min()}")

# Calculate variance
print(f"Variance of convspread: {df_cdsita['convspread'].var()}")
print(f"Variance of bondspread: {df_cdsita['bondspread'].var()}")

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(df_cdsita['convspread'], bins=30, alpha=0.5, label='convspread')
plt.hist(df_cdsita['bondspread'], bins=30, alpha=0.5, label='bondspread')
plt.xlabel('Spread')
plt.ylabel('Frequency')
plt.title('Histogram of Convspread and Bondspread')
plt.legend()
plt.show()

# Analyze the 'basis' variable if it exists
if 'basis' in df_cdsita.columns:
    print(f"Mean of basis: {df_cdsita['basis'].mean()}")
    print(f"Median of basis: {df_cdsita['basis'].median()}")
    print(f"Standard deviation of basis: {df_cdsita['basis'].std()}")
    print(f"Range of basis: {df_cdsita['basis'].max() - df_cdsita['basis'].min()}")
    print(f"Variance of basis: {df_cdsita['basis'].var()}")

    # Plot basis over time
    plt.figure(figsize=(10, 5))
    plt.plot(df_cdsita['basis'], label='Basis')
    plt.xlabel('Date')
    plt.ylabel('Basis')
    plt.title('Basis Over Time')
    plt.legend()
    plt.show()

    # Boxplot of basis
    plt.figure(figsize=(10, 5))
    plt.boxplot(df_cdsita['basis'].dropna(), vert=False)
    plt.xlabel('Basis')
    plt.title('Boxplot of Basis')
    plt.show()
else:
    print("The 'basis' column does not exist in the DataFrame.")
