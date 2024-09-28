# HW1:Linear regression

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

## 過程步驟

![2024-09-28_122249](https://github.com/user-attachments/assets/9f9df792-55cc-4f28-a37c-e26300f9891a)

```bash
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

## 特點

- 功能1：簡要描述
- 功能2：簡要描述
- 功能3：簡要描述

## 安裝

### 使用 Git 克隆

```bash
git clone https://github.com/你的用戶名/專案名稱.git
