import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ----------------------------
# Create Non-linear Dataset
# ----------------------------
np.random.seed(0)
X = np.sort(6 * np.random.rand(100, 1) - 3, axis=0)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

degrees = [1, 2, 3, 5, 10]
results = []

plt.scatter(X, y, color='black')

# ----------------------------
# Train Models for Different Degrees
# ----------------------------
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_error = mean_squared_error(y_train, model.predict(X_train))
    test_error = mean_squared_error(y_test, model.predict(X_test))

    results.append((degree, train_error, test_error))

    X_plot = poly.transform(X)
    y_plot = model.predict(X_plot)

    plt.plot(X, y_plot, label=f"Degree {degree}")

# ----------------------------
# Plot All Models
# ----------------------------
plt.legend()
plt.title("Polynomial Regression & Overfitting")
plt.savefig("polynomial_models.png")
plt.show()

# ----------------------------
# Error Table
# ----------------------------
print("Degree | Train Error | Test Error")
for r in results:
    print(f"{r[0]:>6} | {r[1]:>11.2f} | {r[2]:>10.2f}")
