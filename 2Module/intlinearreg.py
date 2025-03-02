import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

print(df.head())


x = df[['SquareFootage']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training data: {x_train.shape}, {y_train.shape}")
print(f"Testing data: {x_test.shape}, {y_test.shape}")

model = LinearRegression()

model.fit(x_train, y_train)

print(f"Interecept: {model.intercept_}")
print(f"Coefficent: {model.coef_[0]}")


y_pred = model.predict(x_test)

print("Predicted prices:", y_pred)
print("Actual Prices:", y_test.values)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

print(f"R-squared: {r2}")


# Plot the data points
plt.scatter(x_test, y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(x_test, y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Show the plot
plt.show()