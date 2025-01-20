import numpy as np
import math
from typing import Callable, List, Tuple, Dict


from environment_linear import EnvLinear

from environment_kernel_bandit import EnvKernelBandit
import matplotlib.pyplot as plt
import matplotlib


##############################################################################
# Placeholder for your chosen kernel function K(z1, z2):
# For instance, if you use an RBF kernel, or polynomial, or finite-dimensional
# embedding, you can define it here.
##############################################################################

class RewardEval(object):
    def __init__(self, phi, kernel, Zh, lambda_inv, alpha, lamda, Aspace, B, h , H):
        self.Zh = Zh
        self.kernel = kernel
        self.lambda_inv = lambda_inv
        self.alpha = alpha
        self.lamda = lamda
        self.phi = phi
        self.B = B
        self.h = h
        self.H = H
        self.Aspace = Aspace


    def mean_kernel_sample(self, z):
        k_array = np.array([self.kernel(z, zi) for zi in self.Zh])
        result = k_array.dot(self.alpha)
        return result

    def var_kernel_sample(self, z):
        k_array = np.array([self.kernel(z, zi) for zi in self.Zh])
        uncertainty = max(self.kernel(z, z) - np.dot(np.dot(self.lambda_inv, k_array), k_array), 0)
        result = (self.lamda ** 0.5) * ((uncertainty) ** 0.5)
        return result

    def Zhat_h_func(self, z):
        return max(self.mean_kernel_sample(z) - self.B * self.var_kernel_sample(z), 0)

    def Qhat_h_func(self, s, a):
        z = self.phi(s, a)
        result = np.clip(self.Zhat_h_func(z), 0, self.H - self.h)
        return result

    def Vhat_h_func(self, s):
        max_q, _ = max([(self.Qhat_h_func(s, a), a) for a in self.Aspace], key=lambda x: x[0])
        return max_q


class PDSKernel(object):
    def __init__(self, env, phi, kernel, beta1=0.05, beta2=0.05, lamda1=1, lamda2=1):
        self.env = env
        self.phi = phi
        self.kernel = kernel
        self.H = env.H
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        pass


    def build_kernel_matrix(self, Zh, lamda, Rh):
        N1 = len(Zh)
        # Build kernel matrix K of size NxN
        K = np.zeros((N1, N1))
        for i in range(N1):
            zi = Zh[i]  # combine s,a if needed
            for j in range(N1):
                zj = Zh[j]
                #print(f"zi,zj={zi},{zj}")
                K[i, j] = self.kernel(zi, zj)

        lambda_inv = np.linalg.inv(K + lamda * np.eye(len(Zh)))
        alpha = lambda_inv.dot(Rh)
        return K, lambda_inv, alpha

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

    def fit_reward_function(self, D1):
        reward_fn = []

        for h in range(self.env.H):
            Sh, Ah, Rh, Zh = self.data_preprocessing(D1, h)

            K, lambda_inv, alpha = self.build_kernel_matrix(Zh, self.lamda1, Rh)

            lambda_inv = np.linalg.inv(K + self.lamda1 * np.eye(len(Sh)))
            alpha = lambda_inv.dot(Rh)

            reward_fn.append(RewardEval(self.phi, self.kernel, Zh, lambda_inv, alpha, self.lamda1, self.env.A, self.beta1, h, self.env.H))

        return reward_fn



    def relabel_unlabeled_data(self, D1, D2, reward_fn, relabel_D1=True):

        Dtilde = []
        for h in range(self.env.H):
            Dtilde.append([])
            reward_fn_h = reward_fn[h]

            for (s_h_t, a_h_t, r_pess ) in D1[h]:
                if relabel_D1:
                    r_pess = reward_fn_h.Zhat_h_func(self.phi(s_h_t, a_h_t))
                Dtilde[-1].append((s_h_t, a_h_t, r_pess))

            for (s_h_t, a_h_t) in D2[h]:
                r_pess = reward_fn_h.Zhat_h_func(self.phi(s_h_t, a_h_t))
                Dtilde[-1].append((s_h_t, a_h_t, r_pess))

        return Dtilde



    def pevi_kernel_approx(self, Dtheta):

        rl_fn = []

        Sh1 = []
        for h in reversed(range(self.env.H)):

            Sh, Ah, Rh, Zh = self.data_preprocessing(Dtheta, h)

            if len(rl_fn) > 0:
                Rh_p_V = [r + rl_fn[0].Vhat_h_func(s) for r,s in zip(Rh, Sh1)]
            else:
                Rh_p_V = Rh

            K, lambda_inv, alpha = self.build_kernel_matrix(Zh, self.lamda2, Rh_p_V)

            rewad_eval =  RewardEval(self.phi, self.kernel, Zh, lambda_inv, alpha, self.lamda2, self.env.A, self.beta2, h, self.env.H)

            rl_fn.insert(0,rewad_eval)
            Sh1 = Sh


        return rl_fn


    def data_sharing_kernel_approx(self, D1, D2):
        # 1) Learn the reward function \hat{\theta}_h
        reward_fn = self.fit_reward_function(D1)

        ## 2) Relabel unlabeled data D2 with tilde{theta}
        Dtheta = self.relabel_unlabeled_data(D1, D2, reward_fn)

        ## 3) Learn the policy from the relabeled dataset using PEVI (Algorithm 2)
        rl_fn = self.pevi_kernel_approx(Dtheta)

        def pi_reward_hat(h, s):
            _, max_a = max([(reward_fn[h].Qhat_h_func(s, a), a) for a in self.env.A], key=lambda x: x[0])
            return max_a

        def pi_rl_fn(h, s):
            _, max_a  = max([(rl_fn[h].Qhat_h_func(s, a), a) for a in self.env.A], key=lambda x: x[0])
            return max_a

        return  pi_rl_fn, pi_reward_hat

def phi_tuple(s, a):
    z = (s,a)
    return z

def kernel_gaussian(z1, z2, variance=3):
    #normalizing_const = math.sqrt(math.pi / variance)
    return math.exp(- variance * ((z1[0] - z2[0]) ** 2 + (z1[1] - z2[1]) ** 2)) #/ normalizing_const

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


def run_debug_kernel_bandit():
    H = 10
    N1 = 100
    N2 = 10
    env = EnvKernelBandit(s_size=8, H=H)
    env.reset_rng(seed=0)
    D1, D2 = env.gen_dataset(N1=N1, N2=N2, H=H)
    print(env.H)
    pds = PDSKernel(env=env, kernel=kernel_gaussian, phi=phi_tuple)
    pi_bandit_hat, pi_hat = pds.data_sharing_kernel_approx(D1, D2)

    def random_pi(h, s):
        return env.random_pi()
    print("evaluate")
    R1 = evaluate(env=env, pi_func=pi_bandit_hat)
    R2 = evaluate(env=env, pi_func=pi_hat)
    Rrand = evaluate(env=env, pi_func=random_pi)
    print(f"R1={R1}, R2={R2}, Rrand={Rrand}")


def run_debug_linear():
    H = 10
    N1 = 100
    N2 = 10
    env = EnvLinear(s_size=8, H=H)
    env.reset_rng(seed=0)
    D1, D2 = env.gen_dataset(N1=N1, N2=N2, H=H)
    print(env.H)
    pds = PDSKernel(env=env, kernel=kernel_gaussian, phi=phi_tuple)
    pi_bandit_hat, pi_hat = pds.data_sharing_kernel_approx(D1, D2)

    def random_pi(h, s):
        return env.random_pi()
    print("evaluate")
    R1 = evaluate(env=env, pi_func=pi_bandit_hat)
    R2 = evaluate(env=env, pi_func=pi_hat)
    Rrand = evaluate(env=env, pi_func=random_pi)
    print(f"R1={R1}, R2={R2}, Rrand={Rrand}")


if __name__ == "__main__":
    run_debug_kernel_bandit()
    run_debug_linear()
    pass
