from collections import defaultdict
import math

import numpy as np


class EnvKernel(object):
    def __init__(self, H=6, s_size=5, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.s_size = s_size
        self.s_size = s_size
        self.H = H
        self.S = np.eye(self.s_size)
        self.A = np.eye(self.s_size)
        self.R = self.gen_reward()
        self.a_trans = self.gen_a_trans()

        self.alpha = 1.0  # You can tune this


    def gaussian_sampler(self, mu ):
        pass

        exponent = - self.alpha * (s_prime - mu) ** 2

    def P_h(self, s_prime, s, a):
        a_trans = self.rng.permutation(list(range(self.s_size)))
        mu = (a_trans[a] + s) % self.s_size
        exponent = - self.alpha * (s_prime - mu ) ** 2
        numerator = math.exp(exponent)
        return numerator / self.normalizing_const

    def reset_rng(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def gen_init_states(self):
        p = list(reversed(range(self.s_size)))
        p = [x/sum(p) for x in p]
        s = self.S[self.rng.choice(range(self.s_size), p=p)]
        return s

    def gen_reward(self):
        R = np.zeros((self.s_size,self.s_size))
        #for i in range(R.shape[0]):
        R[-1,:]= 1
        R = R.reshape((self.s_size*self.s_size,))
        return R

    def gen_a_trans(self):
        return self.rng.permutation(list(range(self.s_size)))


    def gen_r_sn(self, s,a):
        z = np.matmul(np.array(s)[:,np.newaxis],np.array(a)[np.newaxis,:]).reshape((self.s_size*self.s_size,))
        si = self.rng.choice(range(self.s_size), p=np.dot(self.P,z))
        sn = self.S[si]
        r = np.dot(self.R,z)
        return r, sn

    def random_pi(self):
        return self.A[self.rng.choice(range(self.s_size))]

    def gen_random_trajs(self, N, length, labeled):
        trajs = defaultdict(list)
        for _ in range(N):
            s = self.gen_init_states()
            for i in range(length):
                a = self.random_pi()
                r, sn = self.gen_r_sn(s, a)
                if labeled:
                    trajs[i].append((np.array(s), np.array(a), r))
                else:
                    trajs[i].append((np.array(s), np.array(a)))
                s = sn
        return trajs

    def gen_dataset(self, N1=15, N2=10, H=3):
        D1 = self.gen_random_trajs(N=N1, length=H, labeled = True)
        D2 = self.gen_random_trajs(N=N2, length=H, labeled = False)
        return D1,D2


if __name__ == '__main__':
    env = EnvKernel()
    env.P_h(0.5,1,2)
    pass
