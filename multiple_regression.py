import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Load Dataset
# ----------------------------
data = fetch_california_housing()
X = data.data
y = data.target

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Evaluation Metrics
# ----------------------------
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ----------------------------
# Actual vs Predicted Plot
# ----------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.savefig("actual_vs_predicted.png")
plt.show()

# ----------------------------
# Residual Plot
# ----------------------------
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residuals.png")
plt.show()

# ----------------------------
# Model Coefficients
# ----------------------------
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
