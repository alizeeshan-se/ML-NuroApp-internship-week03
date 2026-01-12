import pickle
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# ----------------------------
# Train Model
# ----------------------------
X, y = fetch_california_housing(return_X_y=True)
model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Save Model using Pickle
# ----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ----------------------------
# Save Model using Joblib
# ----------------------------
joblib.dump(model, "model.joblib")

# ----------------------------
# Save Weights as JSON
# ----------------------------
weights = {
    "intercept": model.intercept_,
    "coefficients": model.coef_.tolist()
}

with open("weights.json", "w") as f:
    json.dump(weights, f)

print("Model saved in pickle, joblib, and JSON formats.")
