#import library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

#sample data
x = np.array([50, 100, 150, 200, 250]).reshape(-1, 1)  # แปลงเป็น array 2D
y = np.array([1.5, 3.0, 4.5, 5.0, 7.5])

#split the data to be training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#build a linear model
model = LinearRegression()
model.fit(x_train, y_train)

#predict
x_pred = model.predict(x_test)

#measure
mse = mean_squared_error(y_test, x_pred)
print(f"MSE: {mse:.2f}")

import matplotlib.pyplot as plt

# plot
plt.scatter(x, y, color="blue", label="Actual Data")

line_x = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)  # ค่าที่จะวาดเส้น
line_y = model.predict(line_x)
plt.plot(line_x, line_y, color="red", label="Prediction Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Linear Regression")
plt.show()

# new data
new_data = np.array([[120],[170],[225]])

#predict new
pre_data = model.predict(new_data)

print(pre_data)
