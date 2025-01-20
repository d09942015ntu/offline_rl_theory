from collections import defaultdict

import numpy as np


class EnvLinear(object): #Refactor done
    def __init__(self, H=6, s_size=10, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.s_size = s_size
        self.H = H
        self.A = list(range(self.s_size))
        self._P = self._gen_transition()
        self._S = np.eye(self.s_size)
        self._A = np.eye(self.s_size)
        self._R = self._gen_reward()


    def _gen_init_states(self):
        p = list(reversed(range(self.s_size)))
        p = [x/sum(p) for x in p]
        _s = self._S[self.rng.choice(range(self.s_size), p=p)]
        return _s

    def _random_pi(self):
        _a =  self._A[self.rng.choice(range(self.s_size))]
        return _a

    def _gen_reward(self):
        R = np.zeros((self.s_size,self.s_size))
        R[-1,:]= 1
        R = R.reshape((self.s_size*self.s_size,))
        return R

    def _phi(self,s,a):
        _s = self._S[s]
        _a = self._A[a]
        _z = np.matmul(_s[:,np.newaxis],_a[np.newaxis,:]).reshape((self.s_size*self.s_size,))
        return _z

    def _gen_transition(self):
        self.reset_rng(self.seed)
        all_p = []
        for i in range(self.s_size):
            pi = self.rng.permutation(np.eye(self.s_size))
            all_p.append(pi)
        P = np.concatenate(all_p,axis=1)
        return P

    def reset_rng(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def gen_init_states(self):
        s = np.argmax(self._gen_init_states())
        return s

    def random_pi(self):
        return np.argmax(self._random_pi())


    def get_r_sn(self, s, a):
        _z = self._phi(s,a)
        sn = self.rng.choice(range(self.s_size), p=np.dot(self._P,_z))
        r = np.dot(self._R,_z)
        return r, sn


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
        self.reset_rng(self.seed)
        D1 = self.gen_random_trajs(N=N1, length=H, labeled = True)
        D2 = self.gen_random_trajs(N=N2, length=H, labeled = False)
        return D1,D2


if __name__ == '__main__':
    env = EnvLinear()
    D1,D2 = env.gen_dataset()
    print(1)
    pass
