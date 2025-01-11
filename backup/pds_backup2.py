import numpy as np
from typing import Callable, List, Tuple, Dict


###############################################################################
# 0. Auxiliary: Example kernel & transformations
###############################################################################

def rbf_kernel(x: np.ndarray, y: np.ndarray, length_scale: float = 1.0) -> float:
    """
    A simple RBF (Gaussian) kernel between two vectors x and y.
    This is just one example; use your own kernel as needed.
    """
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (2.0 * (length_scale ** 2)))


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


###############################################################################
# 1. Kernel Ridge Regression to learn rewards from labeled data
###############################################################################

def kernel_ridge_regression(
        X: List[np.ndarray],
        y: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        lam: float,
        **kernel_kwargs
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Solve the kernel ridge regression problem:
       alpha = argmin_alpha  sum_{i}(y_i - K_row_i^T alpha)^2 + lam * alpha^T K alpha
    Return:
       alpha: (n,) array of solution coefficients
       X    : We keep the training inputs (for re-use in predictions).
    """
    # Build Gram matrix
    K = build_gram_matrix(X, kernel_fn, **kernel_kwargs)
    n = len(X)

    # Solve (K + lam I) alpha = y
    # For numerical stability, you may consider more advanced solvers.
    A = K + lam * np.eye(n)
    alpha = np.linalg.solve(A, y)

    return alpha, X


def predict_krr(
        x_new: np.ndarray,
        alpha: np.ndarray,
        X_train: List[np.ndarray],
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        **kernel_kwargs
) -> float:
    """
    Predict the function value at x_new given:
      alpha (n,) from kernel ridge regression,
      X_train (the training points),
      kernel_fn.
    """
    # k(x_new, X_train):
    kvec = np.array([
        kernel_fn(x_new, x_i, **kernel_kwargs) for x_i in X_train])
    return np.dot(kvec, alpha)


###############################################################################
# 2. Construct the pessimistic reward
###############################################################################

def pessimistic_reward(
        x_new: np.ndarray,
        # The "center" function: we represent it by alpha + training set
        alpha: np.ndarray,
        X_train: List[np.ndarray],
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        penalty: float,
        **kernel_kwargs
) -> float:
    """
    \tilde{r}(x_new) = max(  r_hat(x_new) - penalty * bonus(x_new),  0 ).
    For simplicity, we use a scalar `penalty`.
    In practice, you might compute a data-dependent penalty
    = beta_h(\delta) * norm( (Lambda_h^{D_1})^{-1/2} phi(...) ).
    """
    rhat = predict_krr(x_new, alpha, X_train, kernel_fn, **kernel_kwargs)
    pessimistic_val = rhat - penalty
    return max(pessimistic_val, 0.0)  # ensure non-negativity if r in [0,1]


###############################################################################
# 3. Relabel unlabeled dataset with the pessimistic reward
###############################################################################

def relabel_unlabeled_dataset(
        unlabeled_data: List[Tuple[np.ndarray, float]],
        #    ^ e.g. [(state_action, ? ), ... ]
        #    but you'll have an actual structure: (s, a) etc.
        #    We'll treat them as x in R^d plus a dummy "?" reward.
        alpha_list: Dict[int, np.ndarray],  # alpha for each step h
        X_train_list: Dict[int, List[np.ndarray]],  # training X for each step h
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        pessimism_penalty: Dict[int, float],
        H: int,
        **kernel_kwargs
) -> List[Tuple[np.ndarray, float]]:
    """
    For each (x, ???) in the unlabeled_data, pick the corresponding "step index" h
    and assign the reward using the pessimistic_reward function.
    We'll assume unlabeled_data includes a known step index h.
    """
    relabeled_data = []
    for (x, step_idx) in unlabeled_data:
        # We'll assume step_idx is in [1..H].
        alpha_h = alpha_list[step_idx]
        X_train_h = X_train_list[step_idx]
        penalty_h = pessimism_penalty[step_idx]

        r_new = pessimistic_reward(
            x, alpha_h, X_train_h, kernel_fn,
            penalty=penalty_h, **kernel_kwargs
        )
        relabeled_data.append((x, r_new, step_idx))
    return relabeled_data


###############################################################################
# 4. The PEVI routine (Kernel-based) — using data-splitting
###############################################################################

def pevi_kernel_data_splitting(
        dataset_with_rewards: List[Tuple[np.ndarray, float, int]],
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        H: int,
        B: float = 10.0,
        lam: float = 1.0,
        **kernel_kwargs
):
    """
    Pessimistic Value Iteration with kernel approximation & data splitting.
    For demonstration, we keep it fairly simple.
    The dataset is partitioned into H folds, one for each step h.

    dataset_with_rewards:
       a list of tuples (x, reward, step_idx).
       x is in R^d (representing (s,a)), step_idx in [1..H].

    We produce a naive Q^hat, V^hat, etc.
    In real usage, we'd set up the T operator carefully,
    but let's keep a skeleton version for illustration.

    Returns
    -------
    pi_hat: some representation of the final policy.
    (In a real system, you'd store Q^hat or alpha_h for each step h.)
    """

    # 1. Partition data by step index:
    folds = [[] for _ in range(H + 1)]
    for (x, r, hidx) in dataset_with_rewards:
        folds[hidx].append((x, r))

    # 2. Going backwards h=H,...,1:  build Q^hat_h
    #    We'll illustrate a dummy representation of Q^hat as a dictionary.

    Qhat = {}
    Vhat = {}

    for h in reversed(range(1, H + 1)):
        # gather data for step h
        data_h = folds[h]
        X_h = [pt[0] for pt in data_h]
        Y_h = [pt[1] for pt in data_h]  # reward + V_{h+1}, but we'll do a placeholder

        # In a real system, we do kernel ridge regression:
        #   Q^hat_h( x ) ~ KRR( X_h, Y_h )
        # plus the bonus.
        # Instead, here is just a placeholder array:
        Qhat[h] = {
            'alpha': np.zeros(len(X_h)),  # placeholder
            'Xtrain': X_h,
        }
        # Vhat_h(s) = max_a Qhat_h(s,a), but let's store an array of zeros
        Vhat[h] = np.zeros(len(X_h))

    # 3. Return a naive policy, e.g. pi_hat = greedy wrt Q^hat
    #    We'll just store (Qhat, Vhat).
    pi_hat = (Qhat, Vhat)
    return pi_hat


###############################################################################
# 5. Full Algorithm 1: Putting it all together
###############################################################################

def data_sharing_kernel_approx(
        labeled_data: List[Tuple[np.ndarray, float, int]],
        #   e.g. (x, r, step_idx) for N1 episodes,
        unlabeled_data: List[Tuple[np.ndarray, float]],
        #   e.g. (x, step_idx) for N2 episodes (no reward)
        H: int,
        # hyperparams
        lam_krr: float,
        lam_pevi: float,
        penalty_scale: float,
        B_pevi: float,
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        **kernel_kwargs
):
    """
    Implementation of Algorithm 1 (Data Sharing, Kernel Approximation).

    Steps:
      1) Train reward (r^hat) from labeled_data using KRR
      2) Construct pessimistic reward (tilde{r})
      3) Annotate unlabeled_data with that tilde{r}
      4) Combine D1 and D2^theta => D^theta
      5) Partition D^theta into H folds
      6) Run PEVI with kernel approx => final policy pi_hat

    Parameters
    ----------
    labeled_data : List[(x, r, step_idx)]
        Labeled dataset (states or (state,action) combos with rewards).
    unlabeled_data : List[(x, step_idx)]
        Unlabeled dataset (states or (state,action) combos, no reward).
    H : int
        Horizon length.
    lam_krr : float
        Regularization for Kernel Ridge Regression on rewards.
    lam_pevi : float
        Regularization used inside PEVI procedure.
    penalty_scale : float
        A user-chosen scale for the reward penalty in constructing \tilde{r}.
        The paper uses “beta_h(\delta) * \|(\Lambda_h^{D_1})^{-1/2} \phi\|”.
        Here we simply multiply by penalty_scale as a placeholder.
    B_pevi : float
        The “B” parameter for the PEVI bonus (see Lemma C.3 in the paper).
    kernel_fn : function
        The kernel function k(x, y).
    kernel_kwargs : dict
        Additional arguments to the kernel function (e.g. length_scale).

    Returns
    -------
    pi_hat : object
        A representation of the learned policy from the PEVI algorithm.
    """

    # ----------------------------------------
    # (A)  Learn the reward function from D1
    # ----------------------------------------
    # For each step h in [1..H], gather D1_h
    D1_by_h = [[] for _ in range(H + 1)]
    for (x, r, hidx) in labeled_data:
        D1_by_h[hidx].append((x, r))

    alpha_list = {}
    Xtrain_list = {}

    for h in range(1, H + 1):
        data_h = D1_by_h[h]
        if len(data_h) == 0:
            # no data for step h
            alpha_list[h] = np.array([])
            Xtrain_list[h] = []
            continue

        X_h = [pt[0] for pt in data_h]
        y_h = np.array([pt[1] for pt in data_h])  # rewards

        alpha_h, Xtrain_h = kernel_ridge_regression(
            X_h, y_h,
            kernel_fn=kernel_fn,
            lam=lam_krr,
            **kernel_kwargs
        )
        alpha_list[h] = alpha_h
        Xtrain_list[h] = Xtrain_h

    # ----------------------------------------
    # (B)  Construct the pessimistic reward
    #      tilde{r}_h = max( r_hat_h - penalty,  0 )
    # ----------------------------------------
    # For demonstration, define a penalty for each step h:
    # The paper’s eqn (4.4) uses “beta_h(\delta)*norm(...)”.
    # We do a simpler placeholder:
    pessimism_penalty = {}
    for h in range(1, H + 1):
        # Could be: penalty_scale * Beta_h.
        # We just store penalty_scale for illustration:
        pessimism_penalty[h] = penalty_scale

    # ----------------------------------------
    # (C)  Relabel unlabeled dataset
    # ----------------------------------------
    # unlabeled_data = [ (x, step_idx), ... ]
    unlabeled_data_relab = relabel_unlabeled_dataset(
        unlabeled_data,
        alpha_list,
        Xtrain_list,
        kernel_fn,
        pessimism_penalty,
        H=H,
        **kernel_kwargs
    )

    # ----------------------------------------
    # (D)  Combine labeled + unlabeled => D^theta
    # ----------------------------------------
    # labeled_data  ~ (x, r, h)
    # unlabeled_data_relab ~ (x, r, h)
    # Merge them:
    Dtheta = []
    for (x, r, hidx) in labeled_data:
        Dtheta.append((x, r, hidx))
    for (x, rnew, hidx) in unlabeled_data_relab:
        Dtheta.append((x, rnew, hidx))

    # ----------------------------------------
    # (E)  Partition the combined D^theta into H folds, run PEVI
    #      The paper’s Algorithm 2 does data-splitting:
    #        “Partition D^theta into H disjoint sets...”
    #      We implement a simplified version inside pevi_kernel_data_splitting.
    # ----------------------------------------
    pi_hat = pevi_kernel_data_splitting(
        Dtheta, kernel_fn, H, B_pevi, lam_pevi, **kernel_kwargs
    )

    return pi_hat


###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # Suppose we have:
    #   H = 5
    #   labeled_data with N1=10 examples
    #   unlabeled_data with N2=20 examples
    #   Each example is just a synthetic (x, reward, step_idx).

    H = 5
    labeled_data = []
    for i in range(10):
        x_i = np.random.randn(3)  # e.g. R^3
        r_i = float(np.random.rand())  # reward in [0,1]
        step_i = np.random.randint(1, H + 1)  # step in [1..H]
        labeled_data.append((x_i, r_i, step_i))

    unlabeled_data = []
    for j in range(20):
        x_j = np.random.randn(3)
        step_j = np.random.randint(1, H + 1)
        unlabeled_data.append((x_j, step_j))

    # Hyperparameters for demonstration
    lam_krr = 0.1  # ridge param for reward
    lam_pevi = 1.0  # ridge param for value iteration
    penalty_scale = 0.5  # how big is the penalty in the pessimistic reward?
    B_pevi = 10.0  # the bonus scale for PEVI
    length_scale = 1.0  # RBF kernel parameter

    # Run the full Algorithm 1
    pi_hat = data_sharing_kernel_approx(
        labeled_data,
        unlabeled_data,
        H=H,
        lam_krr=lam_krr,
        lam_pevi=lam_pevi,
        penalty_scale=penalty_scale,
        B_pevi=B_pevi,
        kernel_fn=rbf_kernel,
        length_scale=length_scale
    )

    print("\n=== Learned Policy (Skeleton) ===")
    print("pi_hat =", pi_hat)
