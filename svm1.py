import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Data
data = {
    'Time': ['08:30', '09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30', '16:30', '17:30'] * 5,
    'Level_of_Dust': [0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10, 
    'Voltage': [ 6.71, 6.85, 6.84, 6.75, 6.88, 6.85, 6.65, 6.70, 6.67, 6.37] +
               [ 6.67, 6.81, 6.80, 6.71, 6.85, 6.81, 6.61, 6.67, 6.63, 6.33] +
               [ 6.61, 6.75, 6.74, 6.65, 6.78, 6.75, 6.55, 6.60, 6.57, 6.27] +
               [ 6.59, 6.72, 6.71, 6.62, 6.75, 6.72, 6.52, 6.58, 6.54, 6.25] +
               [ 6.57, 6.71, 6.70, 6.61, 6.74, 6.71, 6.51, 6.57, 6.53, 6.23],
    'Current': [ 0.80, 0.90, 0.92, 1.10, 1.08, 1.08, 0.94, 0.82, 0.64, 0.20] +
               [ 0.76, 0.86, 0.88, 1.05, 1.03, 1.03, 0.90, 0.78, 0.61, 0.19] +
               [ 0.74, 0.83, 0.85, 1.02, 1.00, 1.00, 0.87, 0.76, 0.59, 0.18] +
               [ 0.72, 0.80, 0.82, 0.98, 0.97, 0.97, 0.84, 0.73, 0.57, 0.18] +
               [ 0.66, 0.74, 0.76, 0.90, 0.89, 0.89, 0.77, 0.67, 0.53, 0.16]
}

df = pd.DataFrame(data)

# Convert 'Time' column to float (hours and minutes)
df['Time'] = df['Time'].str.split(':').apply(lambda x: int(x[0]) + int(x[1])/60)

df.fillna(df.mean(), inplace=True)

# Features & Target
X = df[['Time', 'Level_of_Dust']]
y_voltage = df['Voltage']
y_current = df['Current']

# Train-Test Split
X_train, X_test, y_train_v, y_test_v = train_test_split(X, y_voltage, test_size=0.2, random_state=42)
X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_current, test_size=0.2, random_state=42)

# Train Support Vector Machine (SVM) Regression
model_v = SVR(kernel='rbf')  # Radial Basis Function kernel
model_c = SVR(kernel='rbf')  # Radial Basis Function kernel

model_v.fit(X_train, y_train_v)
model_c.fit(X_train, y_train_c)

# Predict
y_pred_v = model_v.predict(X_test)
y_pred_c = model_c.predict(X_test)

# MSE & R² Score
mse_v = mean_squared_error(y_test_v, y_pred_v)
r2_v = r2_score(y_test_v, y_pred_v)

mse_c = mean_squared_error(y_test_c, y_pred_c)
r2_c = r2_score(y_test_c, y_pred_c)

print(f'Voltage - MSE: {mse_v:.4f}, R² Score: {r2_v:.4f}')
print(f'Current - MSE: {mse_c:.4f}, R² Score: {r2_c:.4f}')

# Plotting Actual vs. Predicted Values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Voltage Plot
axes[0].scatter(y_test_v, y_pred_v, color='blue', alpha=0.7)
axes[0].plot([min(y_test_v), max(y_test_v)], [min(y_test_v), max(y_test_v)], 'k--', lw=2)  
axes[0].set_xlabel('Actual Voltage')
axes[0].set_ylabel('Predicted Voltage')
axes[0].set_title('Actual vs. Predicted Voltage')

# Current Plot
axes[1].scatter(y_test_c, y_pred_c, color='red', alpha=0.7)
axes[1].plot([min(y_test_c), max(y_test_c)], [min(y_test_c), max(y_test_c)], 'k--', lw=2)  
axes[1].set_xlabel('Actual Current')
axes[1].set_ylabel('Predicted Current')
axes[1].set_title('Actual vs. Predicted Current')

plt.tight_layout()
plt.show()

# New data for prediction
new_data = {
    'Time': ['08:30', '12:30'],
    'Level_of_Dust': [1.5, 3.4]
}

# New DataFrame
new_df = pd.DataFrame(new_data)

# Convert 'Time' column for new data
new_df['Time'] = new_df['Time'].str.split(':').apply(lambda x: int(x[0]) + int(x[1])/60)

# Predict Voltage & Current
predicted_voltage = model_v.predict(new_df[['Time', 'Level_of_Dust']])
predicted_current = model_c.predict(new_df[['Time', 'Level_of_Dust']])

# Calculate Power (Power = Voltage × Current)
predicted_power = predicted_voltage * predicted_current

# Result
new_df['Predicted Voltage'] = predicted_voltage
new_df['Predicted Current'] = predicted_current
new_df['Predicted Power'] = predicted_power

print(new_df)
