import numpy as np


def gen_d_finite_kernel_function(kernel_params=None):
    """
    Implements a d-finite spectrum kernel.

    Parameters:
    z1, z2 : ndarray
        Input vectors (or matrices). z1 and z2 can be 1D or 2D arrays.
    kernel_params : dict, optional
        Parameters for the kernel. Expected keys:
        - 'd': Number of finite eigenvalues (d-finite spectrum).
        - 'eigenvalues': List of eigenvalues (of size d).
        - 'basis_functions': List of callable basis functions (of size d).

    Returns:
    float or ndarray
        Kernel value(s) between z1 and z2.
    """
    if kernel_params is None:
        raise ValueError("Kernel parameters must be provided.")

    d = kernel_params.get('d')
    eigenvalues = kernel_params.get('eigenvalues')
    basis_functions = kernel_params.get('basis_functions')

    if not (len(eigenvalues) == d and len(basis_functions) == d):
        raise ValueError("Mismatch in the dimensions of eigenvalues or basis functions.")

    def kernel_func(z1,z2) :
        # Ensure z1 and z2 are 2D for consistent handling
        z1 = np.atleast_2d(z1)
        z2 = np.atleast_2d(z2)

        # Initialize the kernel value
        kernel_value = 0

        # Compute the kernel using the finite spectral decomposition
        for i in range(d):
            psi_i_z1 = np.apply_along_axis(basis_functions[i], 1, z1)
            psi_i_z2 = np.apply_along_axis(basis_functions[i], 1, z2)
            kernel_value += eigenvalues[i] * np.dot(psi_i_z1, psi_i_z2.T)

        return kernel_value
    return kernel_func

def gen_d_finite_kernel_function_example():

    eigenvalues = [1.0, 0.8, 0.5, 0.3, 0.1, 0.08]
    d = len(eigenvalues)

    def psi_11(z): return np.sin(np.sum(z))

    def psi_21(z): return np.cos(np.sum(z))

    def psi_12(z): return np.sin(np.sum(2*z))

    def psi_22(z): return np.cos(np.sum(2*z))

    def psi_31(z): return np.sin(np.sum(3*z))

    def psi_32(z): return np.cos(np.sum(3*z))

    #def psi_3(z): return np.sum(z ** 2)

    basis_functions = [psi_11, psi_12, psi_21, psi_22, psi_31, psi_32]

    kernel_params = {
        'd': d,
        'eigenvalues': eigenvalues,
        'basis_functions': basis_functions
    }
    return gen_d_finite_kernel_function(kernel_params)
# Example Usage
if __name__ == "__main__":
    # Define kernel parameters

    z1 = np.array([[1, 2], [3, 4]])
    z2 = np.array([[5, 6], [7, 8]])

    # Compute the kernel matrix
    kernel_func = gen_d_finite_kernel_function_example()
    kernel_matrix = kernel_func(z1,z2)
    print("Kernel Matrix:")
    print(kernel_matrix)
