import numpy as np
from typing import Callable, List, Tuple, Dict

from sympy import Lambda

from kernel_funcs import gen_d_finite_kernel_function_example
from environment_linear import EnvLinear #gen_dataset, gen_r_sn
import matplotlib.pyplot as plt
import matplotlib


##############################################################################
# Placeholder for your chosen kernel function K(z1, z2):
# For instance, if you use an RBF kernel, or polynomial, or finite-dimensional
# embedding, you can define it here.
##############################################################################




def phi(z, dataset=None):
    """
    'Feature map' from z to the RKHS. In many places you don't need to
    explicitly store phi(...) if you do kernel tricks. If you do need
    it explicitly, define it here.
    """
    # In practice, for an infinite-dimensional RKHS, you'd use a kernel trick.
    # For demonstration, let's just return z itself (as if a linear embedding).
    return z


def build_gram_matrix(
        Sh, Ah,
        kernel_fn,
) -> np.ndarray:

    N1 = len(Sh)
    # Build kernel matrix K of size NxN
    K = np.zeros((N1, N1))
    for i in range(N1):
        zi = np.hstack([Sh[i], Ah[i]])  # combine s,a if needed
        for j in range(N1):
            zj = np.hstack([Sh[j], Ah[j]])
            print(f"zi,zj={zi},{zj}")
            K[i, j] = kernel_fn(zi, zj)
    return K, N1

##############################################################################
# Step 1: Kernel Ridge Regression to learn reward function from labeled data
##############################################################################

def fit_reward_function(D1, H, nu):
    #kernel_function, phi_func = gen_d_finite_kernel_function_example()
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
        Zh = np.concatenate([Sh,Ah],axis=1)
        #K, N1 = build_gram_matrix(Sh,Ah,kernel_function)

        ## Solve alpha_h = (K + nu I)^{-1} * Rh
        Ldh = np.matmul(Zh.T,Zh)+ nu *np.eye(Zh.shape[1])
        ## (In practice, might need numerical stability, etc.)
        theta_hat_h = np.dot(np.linalg.inv(Ldh), np.dot(Zh.T,Rh))
        #A = K + nu * np.eye(N1)
        #alpha_h = np.linalg.inv(A).dot(Rh)

        #Z_data = []
        #for i in range(N1):
        #    Z_data.append(np.hstack([Sh[i], Ah[i]]))

        # Save
        theta_hat[h] = theta_hat_h, Ldh

    return theta_hat


##############################################################################
# Step 2: Construct pessimistic reward function parameters:  tilde{theta}_h
##############################################################################

def build_pessimistic_reward(theta_hat, D1, beta_h, H, lambda_operator_dict):
    theta_tilde_fn = [None] * H

    for h in range(H):
        # Retrieve the learned alpha vector for step h
        theta_hat_i, ld_i = theta_hat[h]
        theta_tilde_fn[h] = lambda z : np.dot(theta_hat_i,z) - beta_h * np.dot(np.dot(z,np.linalg.inv(ld_i)),z)**0.5

    return theta_tilde_fn


##############################################################################
# Step 3: Relabel unlabeled dataset D2 using tilde{theta}_h
##############################################################################

def relabel_unlabeled_data(D2, theta_tilde_fn, H):
    D2_tilde = [[] for _ in range(H)]
    #kernel_function, phi_func  = gen_d_finite_kernel_function_example()

    for h in range(H):
        theta_tilde_fn_h = theta_tilde_fn[h]


        # Now for each (s_h^tau, a_h^tau) in D2[h], relabel:
        for (s_h_t, a_h_t) in D2[h]:
            r_pess = theta_tilde_fn_h(np.concatenate([s_h_t, a_h_t]))
            D2_tilde[h].append((s_h_t, a_h_t, r_pess))

    return D2_tilde


##############################################################################
# Step 4: Combine labeled dataset D1 and newly relabeled D2_tilde
##############################################################################

def combine_datasets(D1, D2_tilde, H):
    Dtheta = [[] for _ in range(H)]
    for h in range(H):
        # just concatenate
        Dtheta[h] = D1[h] + D2_tilde[h]
    return Dtheta


##############################################################################
# Step 5: Run the PEVI algorithm with kernel function approximation + data splitting
##############################################################################

def pevi_kernel_approx(Dtheta, H, B, lamda, Aspace):
    #flat_data = []
    #for h_ in range(H):
    #    for row in Dtheta[h_]:
    #        # row is (s_h, a_h, r_h)
    #        flat_data.append((h_, row))

    ## shuffle
    #np.random.shuffle(flat_data)

    ## chunk into H subsets
    #N = len(flat_data)
    #chunk_size = N // H
    #subsets = []
    #idx = 0
    #for h_ in range(H):
    #    subset_h = flat_data[idx: idx + chunk_size]
    #    idx += chunk_size
    #    subsets.append(subset_h)
    ## if there's leftover, you can distribute or ignore for simplicity

    # Step 2: build value iteration from h=H to h=1
    # We'll store Qhat[h] and Vhat[h].
    Qhat = [None] * H
    Vhat = [None] * H

    # Initialize Vhat_{H+1}(.) = 0
    # We'll store a function handle for Vhat_{H+1}(s) = 0 for all s.
    def Vhat_terminal(s):
        return 0.0

    Vhat.append(Vhat_terminal)  # so that Vhat[H] is "terminal"

    Sh1 = []
    for h in reversed(range(H)):

        Sh = []
        Ah = []
        Rh = []
        for (s_h_t, a_h_t, r_h_t) in Dtheta[h]:
            Sh.append(s_h_t)
            Ah.append(a_h_t)
            Rh.append(r_h_t)

        # Convert to np arrays
        Sh = np.array(Sh)
        Ah = np.array(Ah)
        Rh = np.array(Rh)
        Zh = np.concatenate([Sh,Ah],axis=1)

        Ldh = np.matmul(Zh.T, Zh) + nu * np.eye(Zh.shape[1])
        Ldh_inv = np.linalg.inv(Ldh)
        ## (In practice, might need numerical stability, etc.)
        if h == H-1:
            Rh_p_V = Rh

        else:
            Rh_p_V =   [r+Vhat[h+1](s) for r,s in zip(Rh,Sh1)]

        theta_hat_h = np.dot(Ldh_inv, np.dot(Zh.T, Rh_p_V))

        # 2.1: On subset 'subsets[h]', solve the kernel ridge for
        #      widehat{mathcal{B}}_h( widehat{V}_{h+1} ).
        # 2.2: Define bonus Gamma_h, etc.
        # For demonstration, we do a no-op assignment:
        def Qhat_h_func(s,a):
            z = np.concatenate([s,a])
            return np.clip(np.dot(theta_hat_h, z) - B * np.dot(np.dot(z, Ldh_inv), z) ** 0.5,0, H-h)

        Qhat[h] = Qhat_h_func

        def Vhat_h_func(s):
            max_q, _ = max([(Qhat_h_func(s,a),a) for a in Aspace],key=lambda x:x[0])
            return max_q

        Vhat[h] = Vhat_h_func
        Sh1 = Sh

    # Step 3: Construct final policy pi_hat from Qhat
    # For each h, pi_hat_h(a|s) = argmax_a Qhat_h(s,a).
    # We'll define a simple function that returns that greedy action.
    def policy_fn(h, s):

        Qhat_h_func = Qhat[h]
        _, max_a  = max([(Qhat_h_func(s, a), a) for a in Aspace], key=lambda x: x[0])
        return max_a

    pi_hat = policy_fn
    return pi_hat


##############################################################################
# Putting it all together as 'Algorithm 1'
##############################################################################

def data_sharing_kernel_approx(D1, D2,
                               H, beta_h_func, delta, B, nu, lamda, Aspace):
    # 1) Learn the reward function \hat{\theta}_h
    theta_hat = fit_reward_function(D1, H, nu)

    # 2) Construct the pessimistic reward function param tilde{theta}
    #    We need \Lambda_h^{D1}, i.e. the operator sum_{tau} phi(z_h^tau) phi(z_h^tau)^T + nu I
    #    plus an invert. We'll skip the details in this skeleton and pass a mock dictionary:
    lambda_operator_dict = {}
    for h in range(H):
        lambda_operator_dict[h] = None  # placeholder

      #TODO

    theta_tilde_fn = build_pessimistic_reward(theta_hat, D1, beta_h_func, H,
                                           lambda_operator_dict)

    # 3) Relabel unlabeled data D2 with tilde{theta}
    D2_tilde = relabel_unlabeled_data(D2, theta_tilde_fn, H)

    # 4) Combine labeled & unlabeled
    Dtheta = combine_datasets(D1, D2_tilde, H)

    # 5) Learn the policy from the relabeled dataset using PEVI (Algorithm 2)
    pi_hat = pevi_kernel_approx(Dtheta, H, B, lamda, Aspace)

    return pi_hat


def evaluate(env, pi_func):
    R1 = []
    for i in range(100):
        env.reset_rng(i)
        r1 = 0
        sn =env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn = env.gen_r_sn(sn, a)
            r1 += r
        R1.append(r1)
    return np.average(R1)
##############################################################################
# Example main code
##############################################################################
def run_experiment(H, delta, B, nu, lamda, beta_h_func, output_name="default.png"):
    env = EnvLinear(H=H, seed=0)
    def random_pi(h, s):
        return env.random_pi()
    R_rand = evaluate(env=env, pi_func=random_pi)
    for n1 in range(1,5):
        print(f"N1={n1}")
        N2s = []
        R1s = []
        plt.clf()
        plt.figure(figsize=(10, 6))
        for n2 in range(20):
            print(f"N1={n1},N2={n2}")
            N1 = n1*2
            N2 = n2
            env.reset_rng(seed=0)
            D1, D2 = env.gen_dataset(N1=N1,N2=N2,H=H)
            pi_hat = data_sharing_kernel_approx(
                D1, D2,
                env.H,
                beta_h_func, delta,
                B, nu, lamda, env.A
            )
            R1 = evaluate(env=env, pi_func=pi_hat)
            #print(f"N1={N1},N2={N2},R1={:.3f},:.3f}")
            N2s.append(N2)
            R1s.append(R1)
        plt.plot(N2s, R1s, label=f'N1={n1}')
        plt.axhline(y=R_rand, color='r', linestyle='--', label=f'R_rand = {R_rand}')
        plt.xlabel('N2')
        plt.ylabel('R1')
        plt.title('Plot of R1 vs N2 with R_rand Line')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/{output_name}_N1_{n1}.png")

if __name__ == "__main__":
    H = 20

    delta = 0.1
    B = 0.05
    nu = 0.01
    lamda = 1.0
    beta_h_func = 0.05

    run_experiment(H, delta, B, nu, lamda, beta_h_func)

