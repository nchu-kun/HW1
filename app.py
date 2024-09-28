# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Title and description
st.title("Interactive Linear Regression App")
st.write("""
This app lets you generate and visualize a linear regression model with custom parameters.
You can adjust the slope, noise scale, and number of data points.
""")

# Step 3: Add user inputs
st.sidebar.subheader("Model Parameters")

# Slider for slope (a)
a = st.sidebar.slider('Select the slope (a)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)

# Slider for noise scale (c)
c = st.sidebar.slider('Select noise scale (c)', min_value=0.0, max_value=5.0, value=2.0, step=0.1)

# Slider for the number of data points (n)
n = st.sidebar.slider('Select number of data points (n)', min_value=10, max_value=1000, value=100, step=10)

# Step 4: Generate random data based on user inputs
np.random.seed(42)
X = np.random.rand(n, 1) * 10  # Random data for X between 0 and 10
y = a * X + 4 + np.random.randn(n, 1) * c  # Linear relationship with noise

# Step 5: Fit the Linear Regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Step 6: Calculate metrics
mse = mean_squared_error(y, y_pred)
r_squared = model.score(X, y)

# Step 7: Display metrics
st.subheader("Model Performance Metrics")
st.write(f"Slope (a): {a}")
st.write(f"Noise scale (c): {c}")
st.write(f"Number of points (n): {n}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared: {r_squared:.2f}")

# Step 8: Create a DataFrame for visualization
data = pd.DataFrame({
    'X': X.flatten(),
    'Actual y': y.flatten(),
    'Predicted y': y_pred.flatten()
})

# Step 9: Plot the data using Streamlit's chart functionality
st.subheader("Data Visualization")
st.write("Below is a scatter plot of the actual data and a line plot of the predicted data:")

# Use line_chart for visualization
st.line_chart(data[['X', 'Actual y', 'Predicted y']].set_index('X'))

# Optional: Show the data in a table
st.write("Here is a preview of the dataset:")
st.dataframe(data.head())
