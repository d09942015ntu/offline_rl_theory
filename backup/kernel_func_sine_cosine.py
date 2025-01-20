#!/usr/bin/env python3

import numpy as np


def kernel_scalar(x, y):
    """
    Compute the kernel value k(x, y) = cos(2x)cos(2y) + cos(3x)cos(3y) + cos(5x)cos(5y)
                                     + sin(2x)sin(2y) + sin(3x)sin(3y) + sin(5x)sin(5y).
    Parameters
    ----------
    x, y : float
        Scalar inputs.

    Returns
    -------
    float
        Value of the kernel for the inputs x and y.
    """
    return (
            np.cos(2 * x) * np.cos(2 * y)
            + np.cos(3 * x) * np.cos(3 * y)
            + np.cos(5 * x) * np.cos(5 * y)
            + np.sin(2 * x) * np.sin(2 * y)
            + np.sin(3 * x) * np.sin(3 * y)
            + np.sin(5 * x) * np.sin(5 * y)
    )


def kernel_matrix(X, Y):
    """
    Compute the kernel matrix K, where K[i, j] = k(X[i], Y[j]).

    Parameters
    ----------
    X, Y : array-like of shape (n,) and (m,)
        Arrays of 1D points.

    Returns
    -------
    K : ndarray of shape (n, m)
        The kernel matrix.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Reshape X and Y so we can broadcast over both
    X = X[:, np.newaxis]  # shape (n, 1)
    Y = Y[np.newaxis, :]  # shape (1, m)

    # Compute each pairwise contribution
    K = (
            np.cos(2 * X) * np.cos(2 * Y)
            + np.cos(3 * X) * np.cos(3 * Y)
            + np.cos(5 * X) * np.cos(5 * Y)
            + np.sin(2 * X) * np.sin(2 * Y)
            + np.sin(3 * X) * np.sin(3 * Y)
            + np.sin(5 * X) * np.sin(5 * Y)
    )

    return K


def main():
    # Example usage:

    # 1. Scalar version
    x_val = 1.0
    y_val = 2.5
    print(f"k({x_val}, {y_val}) = {kernel_scalar(x_val, y_val)}")

    # 2. Matrix version
    X_vals = np.linspace(0, 2 * np.pi, num=5)  # e.g., [0, 1.57, 3.14, 4.71, 6.28]
    Y_vals = np.linspace(0, 2 * np.pi, num=5)
    K_matrix = kernel_matrix(X_vals, Y_vals)
    print("\nKernel matrix K(X, Y) =\n", K_matrix)


if __name__ == "__main__":
    main()
