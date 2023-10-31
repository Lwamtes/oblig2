import pandas as pd  # For data handling
import numpy as np    # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.linear_model import LinearRegression  # For creating a linear regression model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For model evaluation

# Load the CSV data from the path using Pandas
df = pd.read_csv('TSLA.csv')

# Date in df is stored as a string, so we need to convert it to DateTime input
df['Date'] = pd.to_datetime(df['Date'])

# Define the input (X) and output (Y) for the linear regression model
# Input is Date => X = Date
# Output is price => Y = Close Price
# Linear regression works with numerical features, so we need to convert DateTime to timestamps

X = np.arange(1, len(df) + 1).reshape(-1, 1)
Y = df['Close'].values.reshape(-1, 1)

# Create a linear regression model
linear_regressor = LinearRegression()

# Perform linear regression to fit the model to the data
linear_regressor.fit(X, Y)

# Make predictions based on the fitted model
Y_pred = linear_regressor.predict(X)

# Calculate Mean Absolute Error (MAE) to evaluate model accuracy
mae = mean_absolute_error(Y, Y_pred)

# Calculate Mean Squared Error (MSE) to measure the prediction error
mse = mean_squared_error(Y, Y_pred)

# Calculate R-squared (R²) value to evaluate how well the model fits the data
r2 = r2_score(Y, Y_pred)

# Create a scatterplot to visualize the actual data points and the linear regression line
plt.scatter(X, Y)  # Create a scatter plot of the actual data points
plt.plot(X, Y_pred, color='red')  # Plot the linear regression line in red
plt.xlabel('Date (Days after Initial Date)')  # Set the x-axis label
plt.ylabel('Close Price')  # Set the y-axis label

# Print the MAE, MSE, and R² values to evaluate the model's performance
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Value: {r2:.2f}")

# Show the plot with the data and regression line
plt.show()
