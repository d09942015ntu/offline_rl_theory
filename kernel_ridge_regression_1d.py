import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Generate synthetic data
# ------------------------------
np.random.seed(42)

# True function: y = sin(x) + noise
X_train = np.linspace(-3, 3, 20).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(X_train.shape[0])

# For prediction and plotting:
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

# ------------------------------
# 2. Define RBF (Gaussian) kernel
# ------------------------------
def rbf_kernel(X1, X2, gamma=1.0):
    # Shape: (n1, 1, d), (1, n2, d) -> broadcasting
    X1 = X1.flatten()[:, np.newaxis]
    X2 = X2.flatten()[np.newaxis, :]
    X_diff_square = (X1-X2)**2
    return np.exp(-gamma * X_diff_square)

def rbf_kernel_sample(x,X1,gamma=1.0):
    results = []
    for x1 in X1.flatten():
        results.append(np.exp(-gamma * (x - x1)**2))
    return np.array(results)

# ------------------------------
# 3. Construct kernel matrix K
# ------------------------------
gamma = 0.5           # Hyperparameter for RBF
lambda_reg = 0.1      # Regularization parameter

K = rbf_kernel(X_train, X_train, gamma=gamma)

# ------------------------------
# 4. Solve for alpha
#    alpha = (K + lambda * I)^(-1) * y
# ------------------------------
n = len(X_train)
alpha = np.linalg.inv(K + lambda_reg * np.eye(n)).dot(y_train)

# ------------------------------
# 5. Predict on new points
#    f(x) = sum_i alpha_i * k(x, x_i)
# ------------------------------

y_pred = []
for x_t in X_test.flatten():
    K_test = rbf_kernel_sample(x_t, X_train, gamma=gamma)
    y_pred_i = K_test.dot(alpha)
    y_pred.append(y_pred_i)

# ------------------------------
# 6. Plot results
# ------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.plot(X_test, y_true, 'k--', label='True Function')
plt.plot(X_test, y_pred, 'r', label='KRR Predictions')
plt.title('Kernel Ridge Regression with RBF Kernel')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
print(1)