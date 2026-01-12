import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Create Synthetic Dataset
# ----------------------------
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)

# ----------------------------
# Linear Regression from Scratch
# ----------------------------
class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = 0
        self.b = 0
        self.costs = []

    def mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y):
        n = len(y)

        for _ in range(self.n_iters):
            y_pred = self.w * X + self.b

            dw = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            cost = self.mse(y, y_pred)
            self.costs.append(cost)

    def predict(self, X):
        return self.w * X + self.b

    def r2_score(self, y, y_pred):
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("Weight:", model.w)
print("Bias:", model.b)
print("R2 Score:", model.r2_score(y, y_pred))

# ----------------------------
# Plot Regression Line
# ----------------------------
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title("Linear Regression from Scratch")
plt.savefig("regression_line.png")
plt.show()

# ----------------------------
# Plot Cost Convergence
# ----------------------------
plt.plot(model.costs)
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.savefig("cost_convergence.png")
plt.show()

