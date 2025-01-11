import numpy as np
from typing import Callable, List, Tuple, Dict


##############################################################################
# Placeholder for your chosen kernel function K(z1, z2):
# For instance, if you use an RBF kernel, or polynomial, or finite-dimensional
# embedding, you can define it here.
##############################################################################

def rbf_kernel(x: np.ndarray, y: np.ndarray, length_scale: float = 1.0) -> float:
    """
    A simple RBF (Gaussian) kernel between two vectors x and y.
    This is just one example; use your own kernel as needed.
    """
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (2.0 * (length_scale ** 2)))

def kernel_function(z1, z2, kernel_params=None):
    """
    Compute k(z1, z2), the kernel between two inputs z1, z2.
    z1, z2 are e.g. state-action pairs.
    kernel_params can hold parameters like sigma for RBF kernels, etc.
    """
    # -- Example: trivial linear kernel, just for illustration --
    #   k(z1, z2) = z1.dot(z2)
    # Replace with your real kernel:
    return np.dot(z1, z2)


def phi(z, dataset=None, kernel_params=None):
    """
    'Feature map' from z to the RKHS. In many places you don't need to
    explicitly store phi(...) if you do kernel tricks. If you do need
    it explicitly, define it here.
    """
    # In practice, for an infinite-dimensional RKHS, you'd use a kernel trick.
    # For demonstration, let's just return z itself (as if a linear embedding).
    return z


def build_gram_matrix(
        X: List[np.ndarray],  # list of data points in R^d
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        **kernel_kwargs
) -> np.ndarray:
    """
    Construct the Gram matrix K for data X under the given kernel function.
    K[i,j] = kernel_fn(X[i], X[j]).
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j], **kernel_kwargs)
    return K

##############################################################################
# Step 1: Kernel Ridge Regression to learn reward function from labeled data
##############################################################################

def fit_reward_function(D1, H, nu, kernel_params=None):
    """
    Learn the reward function parameter  (hat{theta}_h)  for  h in [H],
    via the kernel ridge regression:

        hat{theta}_h = arg min_theta  sum_{tau=1 to N1}
                        [ r_h^tau - <phi(s'_h^tau,a'_h^tau), theta> ]^2
                        + nu ||theta||^2_{H_k}

    Parameters
    ----------
    D1 : list of lists
        The labeled dataset. D1[h] is the list of (s,a,r) tuples for step h.
        For example, D1[h] might be an array of shape (N1, 3) if each row is
        (s'_h^tau, a'_h^tau, r_h^tau).
        In practice, you might structure it so that each row is a dictionary
        or a custom object.
    H : int
        Horizon.
    nu : float
        Regularization hyperparameter (in paper, 'nu').
    kernel_params : dict or None
        Additional kernel parameters if needed.

    Returns
    -------
    theta_hat : list of parameters in RKHS
        For h in [1..H], we store the learned parameter vector (or function).
        In an actual kernel-based setting, this might store e.g. dual weights
        rather than a finite-dimensional "theta".
    """
    # We'll store a representation for each horizon step h.
    theta_hat = [None] * H

    for h in range(H):
        # Extract the labeled data for step h from D1
        # Suppose each record is (s_h, a_h, r_h):
        Sh = []
        Ah = []
        Rh = []
        for (s_h_t, a_h_t, r_h_t) in D1[h]:
            Sh.append(s_h_t)
            Ah.append(a_h_t)
            Rh.append(r_h_t)

        # Convert to np arrays
        Sh = np.array(Sh)
        Ah = np.array(Ah)
        Rh = np.array(Rh)

        # In a typical kernel ridge regression approach,
        #   alpha = (K + nu I)^(-1) * R
        #   where K_ij = kernel_function( z_i, z_j ),
        #   and z_i = (s_h^i, a_h^i).
        # For demonstration, let's just store alpha as "theta_hat[h]".

        N1 = len(Rh)
        # Build kernel matrix K of size NxN
        K = np.zeros((N1, N1))
        for i in range(N1):
            zi = np.hstack([Sh[i], Ah[i]])  # combine s,a if needed
            for j in range(N1):
                zj = np.hstack([Sh[j], Ah[j]])
                K[i, j] = kernel_function(zi, zj, kernel_params=kernel_params)

        # Solve alpha_h = (K + nu I)^{-1} * Rh
        # (In practice, might need numerical stability, etc.)
        A = K + nu * np.eye(N1)
        alpha_h = np.linalg.inv(A).dot(Rh)

        # We'll store the alpha vector.  The "true" parameter in the paper
        # is an element of the RKHS, but we can represent it by alpha + data support.
        # So let's store (alpha_h, [z1,...,zN]) to define the function by
        #    r_hat_h(z) = sum_{i=1 to N1} alpha_h[i] * kernel_function(z_i, z).
        Z_data = []
        for i in range(N1):
            Z_data.append(np.hstack([Sh[i], Ah[i]]))

        # Save
        theta_hat[h] = (alpha_h, Z_data)

    return theta_hat


##############################################################################
# Step 2: Construct pessimistic reward function parameters:  tilde{theta}_h
##############################################################################

def build_pessimistic_reward(theta_hat, D1, beta_h, H, lambda_operator_dict):
    """
    Construct the pessimistic reward function from Eqn (6) in the paper:
      tilde{r}_h^{ tilde{theta}_h }(s,a)
        = max{ <hat{theta}_h, phi(s,a)> - beta_h * || Lambda_h^{-1/2} phi(s,a)||,  0 }

    In practice, we implement this by storing the offset "penalty" for each (s,a).
    But strictly, for a general kernel approach, we'd be representing
    tilde{theta}_h in dual form as well.

    Parameters
    ----------
    theta_hat : list
        The list of (alpha_h, Z_data) from fit_reward_function(...).
    D1 : list of lists
        The labeled dataset (for forming the operator Lambda_h^{D1}).
    beta_h : function or list
        If beta_h is constant wrt h, you can provide a float; if it differs by h,
        provide a list or function that returns the radius for each h.
    H : int
        Horizon.
    lambda_operator_dict : dict
        A precomputed dictionary of the form:
          lambda_operator_dict[h] = (Lambda_h^{D1})^-1/2
        if you explicitly form it.  Often in large-scale kernel methods, you do
        not form it directly but rely on approximations.
        This is a placeholder illustrating how you'd apply the penalty.

    Returns
    -------
    theta_tilde : list
        The new "pessimistic" parameters, i.e. each is (alpha_tilde_h, Z_data_tilde).
    """
    theta_tilde = [None] * H

    for h in range(H):
        # Retrieve the learned alpha vector for step h
        alpha_h, Z_data_h = theta_hat[h]

        # For the paper, we define:
        #   tilde{r}_h(s,a) = max{ <hat{theta}_h, phi(s,a)> -
        #                              beta_h * || (Lambda_h^{D1})^-1/2 phi(s,a) || , 0 }
        # We'll store a "dual" representation alpha_tilde that effectively includes
        # that subtracted penalty. But in practice, we often just apply
        #    r_tilde(z) = r_hat(z) - penalty(z).clip(min=0)
        #
        # For illustration, let's store a method that, given (s,a), returns
        # the *pessimistic* predicted reward.

        # Here, we do not do a direct matrix approach to solve for alpha_tilde.
        # Instead, we'll store the function and evaluate on the fly.
        alpha_tilde = (alpha_h, Z_data_h, beta_h(h))  # a placeholder
        theta_tilde[h] = alpha_tilde

    return theta_tilde


##############################################################################
# Step 3: Relabel unlabeled dataset D2 using tilde{theta}_h
##############################################################################

def relabel_unlabeled_data(D2, theta_tilde, H, kernel_params=None):
    """
    Take unlabeled dataset (only s,a pairs, no reward), and fill in reward
    using the pessimistic reward function tilde{r}_h.

    D2 : list of lists
        D2[h] = list of (s_h, a_h) for step h.
    theta_tilde : list
        The list of "pessimistic" parameters for each h, as from build_pessimistic_reward.
    H : int
        Horizon.
    kernel_params : dict
        kernel parameters if needed.

    Returns
    -------
    D2_tilde : list of lists
        Same structure as D2, but each element now is (s_h, a_h, r_tilde_h).
    """
    D2_tilde = [[] for _ in range(H)]

    for h in range(H):
        alpha_tilde_h, Z_data_h, beta_val = theta_tilde[h]
        # alpha_tilde_h was (alpha_h, Z_data_h, someBeta). Let me rename them for clarity:
        alpha_h = alpha_tilde_h

        # We'll define a function to evaluate r_hat_h(s,a). Then we'll subtract penalty.

        def pessimistic_reward_func(s, a):
            # Evaluate <hat{theta}_h, phi(s,a)> in dual form:
            #   r_hat_h(s,a) = sum_{i=1..N1} alpha_h[i] * k( Z_data_h[i], (s,a) )
            z_in = np.hstack([s, a])
            r_hat = 0.0
            for i_i, z_support in enumerate(Z_data_h):
                r_hat += alpha_h[i_i] * kernel_function(z_support, z_in, kernel_params)
            # Then subtract the penalty: beta_val * ||(Lambda_h^{D1})^-1/2 phi(...)||.
            # For demonstration, we do not code the exact norm. We'll do a placeholder:
            penalty = beta_val * 0.5  # placeholder, you'd compute the actual RKHS norm
            r_pess = max(r_hat - penalty, 0.0)
            return r_pess

        # Now for each (s_h^tau, a_h^tau) in D2[h], relabel:
        for (s_h_t, a_h_t) in D2[h]:
            r_pess = pessimistic_reward_func(s_h_t, a_h_t)
            D2_tilde[h].append((s_h_t, a_h_t, r_pess))

    return D2_tilde


##############################################################################
# Step 4: Combine labeled dataset D1 and newly relabeled D2_tilde
##############################################################################

def combine_datasets(D1, D2_tilde, H):
    """
    Combine labeled dataset (D1) and the newly labeled dataset (D2_tilde).
    Return Dtheta = D1 + D2_tilde for each step h.

    D1[h] might look like: [(s,a,r), ... ] for h in [0..H-1]
    D2_tilde[h] might look like: [(s,a,r_tilde), ... ]

    Output Dtheta[h] is the union (concatenation).
    """
    Dtheta = [[] for _ in range(H)]
    for h in range(H):
        # just concatenate
        Dtheta[h] = D1[h] + D2_tilde[h]
    return Dtheta


##############################################################################
# Step 5: Run the PEVI algorithm with kernel function approximation + data splitting
##############################################################################

def pevi_kernel_approx(Dtheta, H, B, lamda, kernel_params=None):
    """
    Implement the kernel-based PEVI with data splitting as described
    in Algorithm 2 (or the version in Appendix) from the paper.

    Here we only give a skeleton:

    1) Randomly split Dtheta into H disjoint subsets of equal size:
         Dtheta = union_{h=1 to H}  of (widetilde{D}_h)
    2) For h = H down to 1:
         - Use (widetilde{D}_h) to estimate   (widehat{mathcal{B}}_h widehat{V}_{h+1})
         - Build the bonus function Gamma_h(...)  = B * ||phi(z)||_{(Lambda^widetilde{D}_h)^{-1}}
         - widehat{Q}_h^... = ...
         - widehat{V}_h^... = ...
    3) Return the final policy pi_hat.

    Note: For large-scale kernel RL, one typically needs approximations
          or a finite feature map. This skeleton simply illustrates the logic.

    """
    # Step 1: Data splitting
    # We suppose Dtheta[h] is the data for step h. Then we flatten it
    # to build a single list and shuffle, then chunk it into H subsets.

    # Flatten all (h-step) data:
    flat_data = []
    for h_ in range(H):
        for row in Dtheta[h_]:
            # row is (s_h, a_h, r_h)
            flat_data.append((h_, row))

    # shuffle
    np.random.shuffle(flat_data)

    # chunk into H subsets
    N = len(flat_data)
    chunk_size = N // H
    subsets = []
    idx = 0
    for h_ in range(H):
        subset_h = flat_data[idx: idx + chunk_size]
        idx += chunk_size
        subsets.append(subset_h)
    # if there's leftover, you can distribute or ignore for simplicity

    # Step 2: build value iteration from h=H to h=1
    # We'll store Qhat[h] and Vhat[h].
    Qhat = [None] * H
    Vhat = [None] * H

    # Initialize Vhat_{H+1}(.) = 0
    # We'll store a function handle for Vhat_{H+1}(s) = 0 for all s.
    def Vhat_terminal(s):
        return 0.0

    Vhat.append(Vhat_terminal)  # so that Vhat[H] is "terminal"

    for h in reversed(range(H)):
        # 2.1: On subset 'subsets[h]', solve the kernel ridge for
        #      widehat{mathcal{B}}_h( widehat{V}_{h+1} ).
        # 2.2: Define bonus Gamma_h, etc.
        # For demonstration, we do a no-op assignment:
        def Qhat_h_func(s, a):
            # Placeholder: Q = (r + Vhat_{h+1}(s')) minus a penalty.
            # In a real code, you'd do
            #   Qhat_h(s,a) = kernel ridge fit on (r + Vhat_{h+1}(next_s)) data
            #                 minus B * norm( ... ), and clipped at [0, H-h+1].
            return 0.0

        Qhat[h] = Qhat_h_func

        def Vhat_h_func(s):
            # argmax wrt a: Qhat_h(s,a)
            # For discrete action space, you'd do something like:
            # best_val = max_{a in A} Qhat_h_func(s,a).
            # If continuous, you might do a separate approach.
            return 0.0

        Vhat[h] = Vhat_h_func

    # Step 3: Construct final policy pi_hat from Qhat
    # For each h, pi_hat_h(a|s) = argmax_a Qhat_h(s,a).
    # We'll define a simple function that returns that greedy action.
    def policy_fn(h, s):
        # in discrete action space:
        #   best_a = argmax_{a} Qhat[h](s,a)
        #   return best_a
        # For demonstration, just return None or some placeholder.
        return None

    pi_hat = policy_fn
    return pi_hat


##############################################################################
# Putting it all together as 'Algorithm 1'
##############################################################################

def data_sharing_kernel_approx(D1, D2,
                               H, beta_h_func, delta, B, nu, lamda,
                               kernel_params=None):
    """
    Implementation of Algorithm 1: Data Sharing, Kernel Approximation.

    Parameters
    ----------
    D1 : list of lists
        Labeled dataset.  D1[h] = list of (s,a,r) for each step h.
    D2 : list of lists
        Unlabeled dataset. D2[h] = list of (s,a) for each step h.
    H : int
        Horizon.
    beta_h_func : function
        A function that returns the beta_h(delta) for each step h, i.e.
        beta_h_func(h) -> float
    delta : float
        Confidence parameter.
    B : float
        Scale for the 'bonus' in the PEVI step.
    nu : float
        Regularization parameter in kernel ridge regression.
    lamda : float
        The parameter 'lambda' (different from 'nu') used in the paper
        for the PEVI step.
    kernel_params : dict or None
        Additional kernel details if needed.

    Returns
    -------
    pi_hat : function
        The learned policy from Algorithm 1.

    Steps:
    1) Fit reward function from D1 (kernel ridge).
    2) Construct pessimistic reward tilde{r} using eqn (6).
    3) Relabel D2 with tilde{r}.
    4) Combine to get D^theta = D1 + D2^tilde.
    5) Run kernel-based PEVI on D^theta to get pi_hat.
    """
    # 1) Learn the reward function \hat{\theta}_h
    theta_hat = fit_reward_function(D1, H, nu, kernel_params=kernel_params)

    # 2) Construct the pessimistic reward function param tilde{theta}
    #    We need \Lambda_h^{D1}, i.e. the operator sum_{tau} phi(z_h^tau) phi(z_h^tau)^T + nu I
    #    plus an invert. We'll skip the details in this skeleton and pass a mock dictionary:
    lambda_operator_dict = {}
    for h in range(H):
        lambda_operator_dict[h] = None  # placeholder

    def beta_h(h_in):
        return beta_h_func(h_in)

    theta_tilde = build_pessimistic_reward(theta_hat, D1, beta_h, H,
                                           lambda_operator_dict)

    # 3) Relabel unlabeled data D2 with tilde{theta}
    D2_tilde = relabel_unlabeled_data(D2, theta_tilde, H, kernel_params=kernel_params)

    # 4) Combine labeled & unlabeled
    Dtheta = combine_datasets(D1, D2_tilde, H)

    # 5) Learn the policy from the relabeled dataset using PEVI (Algorithm 2)
    pi_hat = pevi_kernel_approx(Dtheta, H, B, lamda, kernel_params=kernel_params)

    return pi_hat


##############################################################################
# Example main code
##############################################################################

if __name__ == "__main__":
    # Suppose we have:
    #   H=3 horizon
    #   We have D1 (labeled) and D2 (unlabeled) with a small amount of data.

    # Here, let's just mock them up:
    H = 3

    # D1[h] is a list of (s_h, a_h, r_h)
    # We'll do small arrays for illustration:
    D1 = [[] for _ in range(H)]
    # E.g., for h=0, have a few (s,a,r)
    D1[0] = [
        (np.array([0.1, 0.2]), np.array([0.0]), 0.5),
        (np.array([0.4, 0.1]), np.array([1.0]), 0.7),
    ]
    D1[1] = [
        (np.array([0.9, 0.8]), np.array([0.0]), 0.3),
    ]
    D1[2] = [
        (np.array([0.2, 0.3]), np.array([1.0]), 0.9),
    ]

    # D2[h] is a list of (s_h, a_h) with no r
    D2 = [[] for _ in range(H)]
    D2[0] = [
        (np.array([0.7, 0.2]), np.array([0.0])),
    ]
    D2[1] = []
    D2[2] = [
        (np.array([0.8, 0.8]), np.array([1.0])),
    ]


    # Suppose we define beta_h(delta) = some constant for each h
    def beta_h_func(h_in):
        return 1.0  # placeholder


    delta = 0.1
    B = 2.0
    nu = 0.01
    lamda = 1.0

    # Run the main routine:
    pi_hat = data_sharing_kernel_approx(
        D1, D2,
        H,
        beta_h_func, delta,
        B, nu, lamda,
        kernel_params=None
    )

    # pi_hat is a function that, given (h, s), returns an action (for discrete),
    # or a distribution. Since this is a simple skeleton, pi_hat() just returns None.

    # You now have a policy pi_hat that you can evaluate in your environment.
    print("Learned policy pi_hat is now available.")
