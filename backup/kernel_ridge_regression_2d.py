import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------
# 1. Generate synthetic 2D data
# ------------------------------
np.random.seed(42)

# Define a function of two variables
def true_function(X):
    """
    X is of shape (n_samples, 2) where X[:,0] = x1, X[:,1] = x2.
    We'll define y = sin(x1) * cos(x2).
    """
    return np.sin(X[:, 0]) * np.cos(X[:, 1])

# Create 2D training inputs on a grid, with a bit of noise in the output
grid_size = 10
x1_vals = np.linspace(-3, 3, grid_size)
x2_vals = np.linspace(-3, 3, grid_size)
X1_train, X2_train = np.meshgrid(x1_vals, x2_vals)
X_train = np.column_stack([X1_train.ravel(), X2_train.ravel()])

# True function values + noise
y_train = true_function(X_train) + 0.1 * np.random.randn(X_train.shape[0])

# ------------------------------
# 2. Define the RBF (Gaussian) kernel
# ------------------------------
def rbf_kernel(X1, X2, gamma=0.5):
    """
    Radial Basis Function (RBF) kernel.
    K[i,j] = exp(-gamma * ||x_i - x_j||^2)
    """
    # Expand dimensions for broadcasting
    X1 = X1[:, np.newaxis, :]  # shape (n1, 1, d)
    X2 = X2[np.newaxis, :, :]  # shape (1, n2, d)
    return np.exp(-gamma * np.sum((X1 - X2)**2, axis=2))

# ------------------------------
# 3. Construct the kernel matrix & solve for alpha
#    alpha = (K + lambda * I)^(-1) * y
# ------------------------------
gamma = 0.5           # RBF kernel width parameter
lambda_reg = 0.1      # Regularization
K = rbf_kernel(X_train, X_train, gamma=gamma)
n = X_train.shape[0]
alpha = np.linalg.inv(K + lambda_reg * np.eye(n)).dot(y_train)

# ------------------------------
# 4. Predict on a 2D grid
#    f(x) = sum_i alpha_i * k(x, x_i)
# ------------------------------
test_grid_size = 30
x1_test_vals = np.linspace(-3, 3, test_grid_size)
x2_test_vals = np.linspace(-3, 3, test_grid_size)
X1_test, X2_test = np.meshgrid(x1_test_vals, x2_test_vals)
X_test = np.column_stack([X1_test.ravel(), X2_test.ravel()])

# Construct test-train kernel matrix
K_test = rbf_kernel(X_test, X_train, gamma=gamma)
y_pred = K_test.dot(alpha)

# ------------------------------
# 5. Visualize results
# ------------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# True surface for comparison
y_true = true_function(X_test)
# Reshape for plotting
Y_true_plot = y_true.reshape(test_grid_size, test_grid_size)
Y_pred_plot = y_pred.reshape(test_grid_size, test_grid_size)

# Plot the predicted surface
ax.plot_surface(X1_test, X2_test, Y_pred_plot, cmap='viridis', alpha=0.8, 
                edgecolor='none', label='KRR Predictions')

# Overlay the true surface as wireframe (optional for comparison)
ax.plot_wireframe(X1_test, X2_test, Y_true_plot, color='r', linewidth=0.5,
                  label='True Function')

# Plot training points
# (Note: We set them at their (x, y, z) = (x1, x2, y_train).)
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='black', 
           label='Training data')

ax.set_title('Kernel Ridge Regression on 2D Data')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.tight_layout()
plt.show()