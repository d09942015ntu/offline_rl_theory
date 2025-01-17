from collections import defaultdict

import numpy as np


class EnvLinear(object):
    def __init__(self, H=6, s_size=20, a_size=10, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.s_size = s_size
        self.a_size = a_size
        self.H = H
        self.S = np.eye(self.s_size)
        self.A = np.eye(self.a_size)
        self.R = self.gen_reward()
        self.P = self.gen_transition()

    def reset_rng(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def gen_init_states(self):
        p = list(reversed(range(self.s_size)))
        p = [x/sum(p) for x in p]
        s = self.S[self.rng.choice(range(self.s_size), p=p)]
        return s

    def gen_reward(self):
        R = np.zeros((self.s_size,self.a_size))
        for i in range(R.shape[0]):
            R[i,:]= ((i+1)/R.shape[0])**2
        R = R.reshape((self.s_size*self.a_size,))
        return R

    def gen_transition(self):
        P = np.zeros((self.s_size,self.s_size*self.a_size,))
        for i in range(P.shape[0]):
            for j in range(self.s_size*self.a_size):
                if i*self.a_size == j:
                    P[i,j] = 1
                elif i*self.a_size + 1 == j:
                    P[(i+1)%self.s_size,j] = 0.5
                    P[i,j] = 0.5
                #if self.a_size > 2:
                #    if i * self.a_size + 2 == j:
                #        P[(i + 1) % self.s_size, j] = 0.25
                #        P[(i + 2) % self.s_size, j] = 0.25
                #        P[i, j] = 0.5
                #if self.a_size > 3:
                #    if i * self.a_size + 3 == j:
                #        P[(i + 1) % self.s_size, j] = 0.25
                #        P[(i + 2) % self.s_size, j] = 0.125
                #        P[(i + 3) % self.s_size, j] = 0.125
                #        P[i, j] = 0.5
                #if self.a_size > 4:
                #    if i * self.a_size + 4 == j:
                #        P[(i + 1) % self.s_size, j] = 0.25
                #        P[(i + 2) % self.s_size, j] = 0.125
                #        P[(i + 3) % self.s_size, j] = 0.0625
                #        P[(i + 4) % self.s_size, j] = 0.0625
                #        P[i, j] = 0.5
                for k in range(2, self.a_size):
                    if i * self.a_size + k == j:
                        for l in range(1,k):
                            P[(i + l) % self.s_size, j] = 1/2**(l+1)
                        P[(i + k) % self.s_size, j] = 1/2**(k)
                        P[i, j] = 0.5
        return P


    def get_r_sn(self, s, a):
        z = np.matmul(np.array(s)[:,np.newaxis],np.array(a)[np.newaxis,:]).reshape((self.s_size*self.a_size,))
        si = self.rng.choice(range(self.s_size), p=np.dot(self.P,z))
        sn = self.S[si]
        r = np.dot(self.R,z)
        return r, sn

    def random_pi(self):
        return self.A[self.rng.choice(range(self.a_size))]

    def gen_random_trajs(self, N, length, labeled):
        trajs = defaultdict(list)
        for _ in range(N):
            s = self.gen_init_states()
            for i in range(length):
                a = self.random_pi()
                r, sn = self.get_r_sn(s, a)
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
    pass
