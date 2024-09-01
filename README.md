# Name: SANTHANA LAKSHMI K
# REG NO 212222240091
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
DEVELOPED BY: SANTHANA LAKSHMI K
REG NO 212222240091
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('/content/powerconsumption.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
df = df.sort_values('Datetime')
df['Date_ordinal'] = df['Datetime'].apply(lambda x: x.toordinal())
X = df['Date_ordinal'].values.reshape(-1, 1)
y = df['PowerConsumption_Zone1'].values
```
A - LINEAR TREND ESTIMATION
```
linear_model = LinearRegression()
linear_model.fit(X, y)
df['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(df['Datetime'], df['PowerConsumption_Zone1'], label='Original Data', color='blue')
plt.plot(df['Datetime'], df['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1 (kW)')
plt.legend()
plt.grid(True)
plt.show()
```

B- POLYNOMIAL TREND ESTIMATION
```
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
df['Polynomial_Trend'] = poly_model.predict(X_poly)
plt.figure(figsize=(10, 6))
plt.plot(df['Datetime'], df['PowerConsumption_Zone1'], label='Original Data', color='blue')
plt.plot(df['Datetime'], df['Polynomial_Trend'], color='green', label='Polynomial Trend (Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1 (kW)')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
A - LINEAR TREND ESTIMATION

![image](https://github.com/user-attachments/assets/284e2f89-b825-4b8a-9b84-c977bc68650d)

B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/9c3ddd80-257b-473d-b5e0-ba9dac1dfcfa)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
