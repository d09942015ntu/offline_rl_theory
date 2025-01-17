from collections import defaultdict
import math

import numpy as np

import matplotlib.pyplot as plt

class EnvKernel(object):
    def __init__(self, H=6, s_size=6, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.s_size = s_size
        self.s_size = s_size
        self.H = H
        self.S = np.eye(self.s_size)
        self.A = np.eye(self.s_size)
        self.a_trans = self.gen_a_trans()
        self.s_coeff = 1.0
        self.alpha = 0.2  # You can tune this
        self.num_samples = 1000


    def gaussian_sampler(self, mu, var=1.0 ):
        line_space = np.linspace(mu-0.5*self.s_size, mu+0.5*self.s_size, self.num_samples)
        p = [math.exp(- var * (x - mu ) ** 2) for x in line_space ]
        p = [x/sum(p) for x in p]
        sample = self.rng.choice(line_space,p=p)
        return sample % self.s_size




    def get_sn(self, s, a):
        mu = (self.a_trans[a] + self.s_coeff*s) % self.s_size
        return self.gaussian_sampler(mu, var=0.2)
        #exponent = - self.alpha * (s_prime - mu ) ** 2
        #numerator = math.exp(exponent)
        #return numerator / self.normalizing_const

    def reset_rng(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def gen_init_states(self):
        return self.gaussian_sampler(1, var=0.2)


    def gen_a_trans(self):
        return self.rng.permutation(list(range(self.s_size)))

    def get_r(self, s, a):
        return math.exp(- 0.2 * (s - math.ceil(self.s_size/2) ) ** 2)

    def get_r_sn(self, s, a):
        r = self.get_r(s, a)
        sn = self.get_sn(s, a)
        return r, sn

    def random_pi(self):
        return self.rng.choice(range(self.s_size))

    def gen_random_trajs(self, N, length, labeled):
        trajs = defaultdict(list)
        for _ in range(N):
            s = self.gen_init_states()
            for i in range(length):
                a = self.random_pi()
                r, sn = self.get_r_sn(s, a)
                if labeled:
                    trajs[i].append((s, a, r))
                else:
                    trajs[i].append((s, a))
                s = sn
        return trajs

    def gen_dataset(self, N1=15, N2=10, H=3):
        D1 = self.gen_random_trajs(N=N1, length=H, labeled = True)
        D2 = self.gen_random_trajs(N=N2, length=H, labeled = False)
        return D1,D2


def visualize_distribution(s_primes, fname):
    plt.clf()
    plt.hist(s_primes, bins=30, alpha=0.7, color='blue', edgecolor='black')

    # Add titles and labels
    plt.title('Distribution of s_primes')
    plt.xlabel('s_prime values')
    plt.ylabel('Frequency')

    # Show the plot
    plt.savefig(fname)

def debug_trans():
    env = EnvKernel()
    for s in range(env.s_size):
        print(f"s0={s}")
        for a in range(env.s_size):
            print(f"s,a={s,a}")
            s_primes=[]
            for p in range(200):
                s_prime = env.get_sn(s,a)
                s_primes.append(s_prime)
            visualize_distribution(s_primes, fname=f"results/fig_s_{s}_t_|{"|".join([str(int(x+s*env.s_coeff)%env.s_size) for x in env.a_trans])}|_a_{a}.png")
            print(f"s1={s}")

if __name__ == '__main__':
    env = EnvKernel()
    results = env.gen_dataset()
    print(1)
