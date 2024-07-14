# Options-Pricing-Model

# Options Pricing Models Project

## Overview
Delve into the world of options! This project equips you with Python tools on Google Colab to build and analyze option pricing models, a cornerstone of financial derivatives.

## What You Will Be Learning
- Grasp the fundamentals of option pricing models like the Black-Scholes model.
- Implement these models in Python using libraries like NumPy and SciPy.
- Calculate option prices based on factors like underlying asset price, strike price, volatility, and time to maturity.

## Tools and Libraries
- **Programming Language**: Python
- **Platform**: Google Colab (free Jupyter notebook environment)
- **Libraries**:
  - NumPy: For numerical operations
  - SciPy: For advanced mathematical functions
  - Matplotlib: For data visualization
  - pandas: For data manipulation

## Dataset
- Dataset: [GS Option Prices on Kaggle](https://www.kaggle.com/datasets/mohantys/gs-option-prices)

## Expected Results
The output should include a DataFrame displaying bid prices and the calculated option prices based on the Black-Scholes model. Additionally, the project should plot the bid prices against the modeled prices.

## How to Run the Project
1. **Setup Environment**: Install necessary libraries.
2. **Upload and Extract Dataset**: Handle the dataset file from Kaggle.
3. **Load Dataset**: Read the extracted CSV file into a pandas DataFrame.
4. **Extract Relevant Parameters**: Get the necessary columns for the Black-Scholes model.
5. **Calculate Option Prices**: Use the Black-Scholes model to compute option prices.
6. **Create DataFrame**: Combine bid prices and calculated option prices into a DataFrame.
7. **Display and Plot Results**: Print the DataFrame and plot the prices for visualization.

## Raw Code
```python
# Setup Environment
!pip install numpy scipy matplotlib pandas

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Function for Black-Scholes model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Load dataset from Kaggle
# Note: Make sure to upload the zip file containing the dataset to your Google Colab environment
from google.colab import files
uploaded = files.upload()

import zipfile
import os

# Define the zip file name and the directory to extract to
zip_file_name = 'gs-option-prices.zip'
extract_dir = '/content'

# Extract the zip file
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    
# List the extracted files
extracted_files = os.listdir(extract_dir)
print(extracted_files)

# Assuming the CSV file is named 'GS_option_prices.csv'
csv_file_path = os.path.join(extract_dir, 'GS_option_prices.csv')
df = pd.read_csv(csv_file_path)

# Display the first few rows to understand the structure
print(df.head())

# Extract relevant parameters from the dataset
# Assuming the columns are named as 'Underlying_Price', 'Strike_Price', 'Time_to_Maturity', 'Risk_Free_Rate', 'Volatility'
S = df['Underlying_Price']
K = df['Strike_Price']
T = df['Time_to_Maturity']
r = df['Risk_Free_Rate']
sigma = df['Volatility']

# Calculate modeled prices using the Black-Scholes model
modeled_prices = [black_scholes(S[i], K[i], T[i], r[i], sigma[i], option_type='call') for i in range(len(df))]

# Create DataFrame to display Bid prices and Modeled Prices
result_df = pd.DataFrame({
    'Bid': S,
    'm_prices': modeled_prices
})

# Display DataFrame
print(result_df.head())

# Plot the prices if necessary
plt.figure(figsize=(10, 6))
plt.plot(S, modeled_prices, 'o-', label='Modeled Prices')
plt.xlabel('Bid Prices')
plt.ylabel('Modeled Prices')
plt.title('Bid Prices vs. Modeled Prices')
plt.legend()
plt.show()
