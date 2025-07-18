from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

x = 90

# Load global model from FL round 5
model = load_model("./model/global_model_round_5.h5")

# Define test inputs
x_vals = np.linspace(0, 100, 100).reshape(-1, 1)

# Predict using global model
y_pred = model.predict(x_vals)

# Actual ground truth values based on y = 2x + 1
y_actual = 2 * x_vals + 1

# Plot actual vs predicted
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_actual, label="Actual (y = 2x + 1)", color="green", linestyle="--")
plt.plot(x_vals, y_pred, label="Predicted (Global Model)", color="blue")
plt.scatter([90], [181], color="red", label="Expected y at x=90")

# Formatting
plt.xlabel("x")
plt.ylabel("y")
plt.title("Global Model: Actual vs Predicted")
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig("./plots/global_model_vs_actual.png")
plt.show()

# Predict for a single test value (e.g., x=150)
x_test = np.array([[x]], dtype=np.float32)
y_test_pred = model.predict(x_test)
print(f"📈 Prediction for x={x_test[0][0]}: y={y_test_pred[0][0]}")
