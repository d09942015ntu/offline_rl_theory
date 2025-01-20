import numpy as np
import math
from typing import Callable, List, Tuple, Dict

from sympy import Lambda

from kernel_funcs import gen_d_finite_kernel_function_example
from environment_linear_old import EnvLinearOld #gen_dataset, gen_r_sn
from environment_linear import EnvLinear
from environment_kernel_old import EnvKernel

from environment_kernel_bandit import EnvKernelBandit
import matplotlib.pyplot as plt
import matplotlib


##############################################################################
# Placeholder for your chosen kernel function K(z1, z2):
# For instance, if you use an RBF kernel, or polynomial, or finite-dimensional
# embedding, you can define it here.
##############################################################################




class PDS(object):
    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self.H = env.H
        pass


    def phi(self, s, a):
        z = (s,a)
        return z

    def build_gram_matrix(self, Zh) -> np.ndarray:
        N1 = len(Zh)
        # Build kernel matrix K of size NxN
        K = np.zeros((N1, N1))
        for i in range(N1):
            zi = Zh[i]  # combine s,a if needed
            for j in range(N1):
                zj = Zh[j]
                #print(f"zi,zj={zi},{zj}")
                K[i, j] = self.kernel(zi, zj)
        return K

    def data_preprocessing(self, D, h):
        Sh = []
        Ah = []
        Rh = []
        for (s_h_t, a_h_t, r_h_t) in D[h]:
            Sh.append(s_h_t)
            Ah.append(a_h_t)
            Rh.append(r_h_t)
        Sh = np.array(Sh)
        Ah = np.array(Ah)
        Rh = np.array(Rh)
        Zh = np.array([self.phi(s, a) for s, a in zip(Sh, Ah)])
        return Sh, Ah, Rh, Zh

    def fit_reward_function(self, D1, nu):
        H = self.env.H
        reward_fn = [None] * H

        for h in range(H):
            Sh, Ah, Rh, Zh = self.data_preprocessing(D1, h)

            K = self.build_gram_matrix(Zh)

            lambda_inv = np.linalg.inv(K + nu * np.eye(len(Sh)))
            alpha = lambda_inv.dot(Rh)

            def mean_kernel_sample(z):
                k_array = np.array([self.kernel(z,zi) for zi in Zh])
                return k_array.dot(alpha)

            def var_kernel_sample(z):
                k_array = np.array([self.kernel(z,zi) for zi in Zh])
                return (nu**0.5)*((self.kernel(z,z) - np.dot(np.dot(lambda_inv,k_array),k_array))**0.5)


            reward_fn[h] = mean_kernel_sample, var_kernel_sample

        return reward_fn


    def build_pessimistic_reward(self, theta_hat, beta_h):
        theta_tilde_fn = [None] * self.env.H

        for h in range(self.env.H):
            mean_kernel_sample, var_kernel_sample = theta_hat[h]
            theta_tilde_fn[h] = lambda z : mean_kernel_sample(z) - beta_h * var_kernel_sample(z)

        return theta_tilde_fn


    def relabel_unlabeled_data(self, D1, D2, reward_fn, beta_h_func):

        theta_tilde_fn = self.build_pessimistic_reward(reward_fn, beta_h_func)

        D2_tilde = [[] for _ in range(self.env.H)]

        for h in range(self.env.H):
            theta_tilde_fn_h = theta_tilde_fn[h]
            for (s_h_t, a_h_t) in D2[h]:
                r_pess = theta_tilde_fn_h(self.phi(s_h_t, a_h_t))
                D2_tilde[h].append((s_h_t, a_h_t, r_pess))

        Dtheta = self.combine_datasets(D1, D2_tilde)

        return Dtheta


    def combine_datasets(self, D1, D2_tilde):
        Dtheta = [[] for _ in range(self.env.H)]
        for h in range(self.env.H):
            Dtheta[h] = D1[h] + D2_tilde[h]
        return Dtheta


    def pevi_kernel_approx(self, Dtheta, B, lamda):
        H = self.env.H
        Aspace = self.env.A

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

            Sh, Ah, Rh, Zh = self.data_preprocessing(Dtheta, h)

            K = self.build_gram_matrix(Zh)

            if h == H-1:
                Rh_p_V = Rh

            else:
                Rh_p_V =   [r+Vhat[h+1](s) for r,s in zip(Rh,Sh1)]

            lambda_inv = np.linalg.inv(K + lamda * np.eye(len(Sh)))
            alpha = lambda_inv.dot(Rh_p_V)

            def mean_kernel_sample(z):
                k_array = np.array([self.kernel(z,zi) for zi in Zh])
                return k_array.dot(alpha)

            def var_kernel_sample(z):
                k_array = np.array([self.kernel(z,zi) for zi in Zh])
                return (lamda**0.5)*((self.kernel(z,z) - np.dot(np.dot(lambda_inv,k_array),k_array))**0.5)

            def Qhat_h_func(s,a):
                z = self.phi(s, a)
                return np.clip(mean_kernel_sample(z) - B * var_kernel_sample(z), 0, H-h)


            Qhat[h] = Qhat_h_func

            def Vhat_h_func(s):
                max_q, _ = max([(Qhat_h_func(s,a),a) for a in Aspace],key=lambda x:x[0])
                return max_q

            Vhat[h] = Vhat_h_func
            Sh1 = Sh

        def policy_fn(h, s):
            Qhat_h_func = Qhat[h]
            _, max_a  = max([(Qhat_h_func(s, a), a) for a in Aspace], key=lambda x: x[0])
            return max_a

        pi_hat = policy_fn
        return pi_hat


    ##############################################################################
    # Putting it all together as 'Algorithm 1'
    ##############################################################################

    def data_sharing_kernel_approx(self, D1, D2, beta_h_func,  B, nu, lamda):
        # 1) Learn the reward function \hat{\theta}_h
        reward_fn = self.fit_reward_function(D1, nu)

        theta_tilde_fn = self.build_pessimistic_reward(reward_fn, beta_h_func)

        def policy_fn(h, s):
            _, max_a = max([(theta_tilde_fn[h](self.phi(s, a)), a) for a in self.env.A], key=lambda x: x[0])
            return max_a

        ## 2) Relabel unlabeled data D2 with tilde{theta}
        Dtheta = self.relabel_unlabeled_data(D1, D2, reward_fn, beta_h_func)

        ## 3) Learn the policy from the relabeled dataset using PEVI (Algorithm 2)
        pi_hat = self.pevi_kernel_approx(Dtheta, B, lamda)

        return policy_fn


def kernel(z1, z2, variance=3):
    normalizing_const = math.sqrt(math.pi / variance)
    return math.exp(- variance * ((z1[0] - z2[0]) ** 2 + (z1[1] - z2[1]) ** 2)) / normalizing_const

def evaluate(env, pi_func):
    R1 = []
    for i in range(100):
        env.reset_rng(i)
        r1 = 0
        sn =env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn = env.get_r_sn(sn, a)
            r1 += r
        R1.append(r1)
    return np.average(R1)
##############################################################################



def run_debug():
    H = 6
    delta = 0.1
    B = 0.05
    beta_h_func = 0.05
    lamda = 1
    nu = 1

    N1=100
    N2=10
    env = EnvKernelBandit(s_size=8, H=H)
    print(f"env.R={env._R}")
    env.reset_rng(seed=0)
    D1, D2 = env.gen_dataset(N1=N1, N2=N2, H=H)
    print(env.H)
    pds = PDS(env=env,kernel=kernel)
    pi_hat = pds.data_sharing_kernel_approx( D1, D2, beta_h_func, B, nu, lamda)

    def random_pi(h, s):
        return env.random_pi()
    R1 = evaluate(env=env, pi_func=pi_hat)
    Rrand = evaluate(env=env, pi_func=random_pi)
    print(f"R1={R1}, Rrand={Rrand}")


if __name__ == "__main__":
    run_debug()
    pass
