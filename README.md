import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 生成示例數據
# X 為自變量，y 為因變量
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 隨機生成 100 個點
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + 隨機噪聲

# 將數據集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建線性回歸模型
model = LinearRegression()

# 擬合模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 繪製結果
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# 輸出模型的係數和截距
print(f'Coefficients: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
