import numpy as np


def compute_suboptimality(d, N1, N2, H, cprime,
                          beta_h_delta, B):
    """
    Compute the suboptimality bound from Corollary 5.2 in the d-finite spectrum case.

    Parameters
    ----------
    d : int
        The (effective) dimension for the kernel with d-finite spectrum.
    N1 : int
        Number of labeled trajectories.
    N2 : int
        Number of unlabeled trajectories.
    H : int
        Horizon length.
    cprime : float
        Constant related to coverage assumption (Assumption 4.2).
    beta_h_delta : float
        The term beta_h(delta) that appears in the corollary (reward uncertainty).
    B : float
        The Bellman-pessimism bound (value uncertainty).

    Returns
    -------
    subopt : float
        The suboptimality bound:
           2 * beta_h_delta * H * cprime / sqrt(N1)
           + 2 * B * H * cprime / sqrt(N / H).
    """

    # total # of trajectories
    N = N1 + N2

    # Each of the H folds has size N_h = N / H
    N_h = N / H

    # The two main terms from the corollary:
    term_reward = 2.0 * beta_h_delta * H * cprime / np.sqrt(N1)
    term_value = 2.0 * B * H * cprime / np.sqrt(N_h)

    # Summation of the two terms:
    subopt = term_reward + term_value
    return subopt


if __name__ == "__main__":
    # Example usage:

    # -------------------------------
    # Step 1: Provide known parameters
    # (In practice, you'd have these from your setting or proofs.)

    d = 20  # effective dimension (d-finite spectrum)
    N1 = 1000  # # labeled trajectories
    N2 = 5000  # # unlabeled trajectories
    H = 10  # horizon length

    # cprime is typically sqrt(2 / c_min) from the coverage analysis,
    # here we just give a placeholder:
    cprime = 1.0

    # Suppose we have derived (or chosen) beta_h(delta) for the reward function
    # from the kernel ridge regression theory:
    beta_h_delta = 5.0  # placeholder example

    # Suppose we have derived B from the PEVI bonus theory:
    B = 30.0  # placeholder example

    # -------------------------------
    # Step 2: Calculate suboptimality
    subopt_bound = compute_suboptimality(
        d, N1, N2, H, cprime, beta_h_delta, B
    )

    # -------------------------------
    # Step 3: Print result
    print(f"Suboptimality bound from Corollary 5.2: {subopt_bound:.4f}")
