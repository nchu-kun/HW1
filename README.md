# HW1 : Linear regression

## https://kunhw1.streamlit.app/

![e5b29c28-5024-45ff-bddd-c4777edfea15](https://github.com/user-attachments/assets/f5e98cef-a4ed-4b7a-a2c2-b6d9a41dc6bd)

## 業務理解：
- 定義我們要建立線性迴歸模型來預測數據的關係，並簡要介紹該模型的應用。
## 數據理解：
- 用戶可以動態設置數據的斜率、噪音比例、和數據點數量，並且系統會生成對應的數據集。
## 數據準備：
- 將數據集分為訓練集和測試集，以便進行模型訓練和評估。
## 建模：
- 基於訓練數據使用 LinearRegression 模型來進行線性迴歸分析。
## 評估：
- 計算並顯示模型的訓練和測試集上的均方誤差（MSE）和決定係數（R-squared）。
## 部署：
- 展示結果，包括生成的數據集和模型的預測結果。

## 第一版程式

- ChatGPT：

![2024-09-28_122249](https://github.com/user-attachments/assets/9f9df792-55cc-4f28-a37c-e26300f9891a)

- 程式碼：

```python
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Generate some random data (or load your data)
# For simplicity, let's assume a linear relationship: y = 3x + 4 + noise
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 random points for X between 0 and 10
y = 3 * X + 4 + np.random.randn(100, 1) * 2  # Adding some noise

# Step 3: Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Make predictions
y_pred = model.predict(X)

# Step 5: Evaluate the model
mse = mean_squared_error(y, y_pred)
r_squared = model.score(X, y)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")

# Optional: Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

- 演示結果：

![2024-09-28_123858](https://github.com/user-attachments/assets/73eba410-6f37-487d-b9a1-55f45eb1e5b0)


## 第二版程式

- ChatGPT：
  
![2024-09-28_122617](https://github.com/user-attachments/assets/183b09d5-7afc-4905-a959-323d17752028)

- 程式碼：

```python
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Title and description
st.title("Simple Linear Regression App")
st.write("""
This app performs a simple linear regression on generated data.
""")

# Step 3: Generate random data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Random data for X
y = 3 * X + 4 + np.random.randn(100, 1) * 2  # Linear relationship with noise

# Step 4: Fit the Linear Regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Step 5: Calculate metrics
mse = mean_squared_error(y, y_pred)
r_squared = model.score(X, y)

# Step 6: Display metrics
st.subheader("Model Performance Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared: {r_squared:.2f}")

# Step 7: Create a DataFrame for visualization
data = pd.DataFrame({
    'X': X.flatten(),
    'Actual y': y.flatten(),
    'Predicted y': y_pred.flatten()
})

# Step 8: Plot the data using Streamlit's chart functionality
st.subheader("Data Visualization")
st.write("Below is a scatter plot of the actual data and a line plot of the predicted data:")

# Streamlit's line_chart can be used for quick visualization
st.line_chart(data[['X', 'Actual y', 'Predicted y']].set_index('X'))

# Optional: Use Streamlit's data display
st.write("Here is a preview of the dataset:")
st.dataframe(data.head())
```

- 演示結果：
  
![2024-09-28_124944](https://github.com/user-attachments/assets/8d00a93a-41dd-43d0-a485-0a4015b94871)



## 最終版程式

- ChatGPT：
  
![2024-09-28_122711](https://github.com/user-attachments/assets/0d17fd60-9991-4cff-9570-c9c42a61ef6a)



- 程式碼：

```python
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
```

- 演示結果：
  
![e5b29c28-5024-45ff-bddd-c4777edfea15](https://github.com/user-attachments/assets/f5e98cef-a4ed-4b7a-a2c2-b6d9a41dc6bd)
