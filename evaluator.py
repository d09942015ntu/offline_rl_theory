import numpy as np
import json

from environment_linear import EnvLinear
from environment_kernel_bandit import EnvKernelBandit
from environment_kernel import EnvKernel
from pds_kernel import PDSKernel, kernel_gaussian, phi_tuple
import matplotlib.pyplot as plt
##############################################################################


def evaluate(env, pi_func):
    r1s = []
    for i in range(100):
        env.reset_rng(i)
        r1 = 0
        sn =env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn = env.get_r_sn(sn, a)
            r1 += r
        r1s.append(r1)
    return np.average(r1s)

def env_experiments(env, pds, n1, n2, H):
    D1, D2 = env.gen_dataset(N1=n1, N2=n2, H=H)
    pi_hat, pi_bandit_hat= pds.data_sharing_kernel_approx(D1, D2)
    r1 = evaluate(env=env, pi_func=pi_hat)
    print(f"N1={n1}, N2={n2}, R1={r1}")
    return r1

def plot_result(n2s, r1s, r_rand, fname):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(n2s, r1s)
    plt.axhline(y=r_rand, color='r', linestyle='--', label=f'R_rand = {r_rand}')
    plt.xlabel('N2')
    plt.ylabel('R1')
    plt.title('Plot of R1 vs N2 with R_rand Line')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    json.dump({'n2':n2s, 'r1':r1s, 'rrand':r_rand},open(fname.replace(".png",".json"),"w"), indent=2 )


def run():
    H = 8
    s_size = 8
    envs = {
            'kernel' : EnvKernel(s_size=s_size,H=H),
            'kernel_bandit': EnvKernelBandit(s_size=s_size, H=H),
            'linear': EnvLinear(s_size=s_size, H=H),
            }

    n1s = [10, 20, 50, 100]
    n2s = [0, 20, 50, 100, 200, 500, 1000]

    for ekey, env in envs.items():
        pds = PDSKernel(env=env, kernel=kernel_gaussian, phi=phi_tuple)

        for n1 in n1s: #,100]:
            def random_pi(h, s):
                return env.random_pi()
            r_rand = evaluate(env=env, pi_func=random_pi)
            plt.clf()
            plt.figure(figsize=(10, 6))
            r1s = []
            for n2 in n2s:
                r1 = env_experiments(env, pds, n1, n2, H)
                r1s.append(r1)
            fname = f"results/env_{ekey}_n1_{n1}.png"
            plot_result(n2s, r1s, r_rand, fname)

if __name__ == "__main__":
    run()
