import os

import numpy as np
import json
import argparse

from environment_linear import EnvLinear
from environment_kernel_bandit import EnvKernelBandit
from environment_kernel import EnvKernel
from environment_kernel2 import EnvKernel2
from environment_carpole import EnvCarpole
from environment_fzlake import EnvFrozenLake
from pds_kernel import (PDSKernel, kernel_gaussian, kernel_linear,
                        phi_tuple, phi_array, phi_linear_2, phi_quadratic_2, phi_tabular_64_4, phi_array_64_4)
import matplotlib.pyplot as plt
##############################################################################


def evaluate(env, pi_func, repeat):
    r1s = []
    for i in range(repeat):
        env.reset_rng(i)
        r1 = 0
        sn =env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn, done = env.get_r_sn(sn, a)
            r1 += r
            if done:
                break
        r1s.append(r1)
    return np.average(r1s)

def env_experiments(env, pds, n1, n2, H, repeat):
    r1s = []
    for i in range(repeat):
        env.reset_rng(i)
        D1, D2 = env.gen_dataset(N1=n1, N2=n2, H=H)
        pi_hat, pi_bandit_hat= pds.data_sharing_kernel_approx(D1, D2)
        r1 = evaluate(env=env, pi_func=pi_hat, repeat=repeat)
        r1s.append(r1)
        print(f"Repeat={i}: N1={n1}, N2={n2}, R1={r1}")
    r1 = np.average(r1s)
    print(f"Average: N1={n1}, N2={n2}, R1={r1}")
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


def run(n1s,n2s, arg_kernel='kernel_gaussian',arg_phi='phi_array', output_dir="results", repeat=1):
    arg_str=f"n1s={n1s},n2s={n2s}, arg_kernel={arg_kernel},arg_phi={arg_phi}, output_dir={output_dir}, repeat={repeat}"
    print(arg_str)
    H = 100
    envs = {
            'carpole' : EnvFrozenLake(H=H),
            #'kernel_bandit': EnvKernelBandit(s_size=s_size, H=H),
            #'linear': EnvLinear(s_size=s_size, H=H),
            }

    #n1s = [10,20,50,100,200]
    #n2s = [10,20,50,100,200,500]
    if arg_kernel == 'kernel_gaussian':
        kernel = kernel_gaussian
    elif arg_kernel == 'kernel_linear':
        kernel = kernel_linear
    else:
        kernel = kernel_gaussian

    if arg_phi == 'phi_array':
        phi = phi_array_64_4
    elif arg_phi == 'phi_tabular':
        phi = phi_tabular_64_4
    else:
        phi = phi_array

    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir,"args.txt"),"w")
    f.write(arg_str)
    f.close()

    for ekey, env in envs.items():

        pds = PDSKernel(env=env, kernel=kernel, phi=phi)

        for n1 in n1s: #,100]:
            def random_pi(h, s):
                return env.random_pi()
            r_rand = evaluate(env=env, pi_func=random_pi, repeat=repeat)
            print(f"r_rand={r_rand}")
            plt.clf()
            plt.figure(figsize=(10, 6))
            r1s = []
            for n2 in n2s:
                r1 = env_experiments(env, pds, n1, n2, H, repeat)
                r1s.append(r1)

            fname = f"{output_dir}/env_{ekey}_n1_{n1}.png"
            plot_result(n2s, r1s, r_rand, fname)



def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument('--kernel', type=str,  choices=['kernel_linear', 'kernel_gaussian'], default='kernel_linear')
    parser.add_argument('--phi', type=str,  choices=['phi_array', 'phi_tabular'], default='phi_tabular')
    parser.add_argument('--output_dir', type=str,  default='results')
    parser.add_argument('--repeat', type=int,  default=1)
    parser.add_argument('--n1s', type=int, nargs='+', default=[500])
    parser.add_argument('--n2s', type=int, nargs='+', default=[100,200,500,1000,2000])

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with arguments
    #n1s = [20,50,100,200][:args.n1s]
    #n2s = [10,20,50,100,200,500][:args.n2s]

    #n1s = [200,500,1000][:args.n1s]
    #n2s = [100,200,500,1000,2000][:args.n2s]

    run(args.n1s, args.n2s, args.kernel, args.phi, args.output_dir, args.repeat)

if __name__ == "__main__":
    main()

