from collections import defaultdict

import numpy as np


def gen_r_sn(s,a,rng):
    z = np.matmul(np.array(s)[:,np.newaxis],np.array(a)[np.newaxis,:]).reshape((6,))
    R = np.array([[0.1, 0.1, 0.4, 0.4, 0.8, 0.8]])
    #10010, 10001, 01010, 01001, 0011, 00101
    P = [
        [1.0, 0.5,   0.0,   0.0,   0.0,   0.5],
        [0.0, 0.5,   1.0,   0.5,   0.0,   0.0],
        [0.0, 0.0,   0.0,   0.5,   1.0,   0.5],
    ]
    si = rng.choice([0,1,2], p=np.dot(P,z))
    sn=[[1, 0, 0], [0, 1, 0], [0, 0, 1]][si]
    r = np.dot(R,z)[0]
    return r, sn

def random_pi(rng):
    return [[1,0],[0,1]][rng.choice([0,1])]

def gen_random_trajs(N, length, labeled, rng):
    trajs = defaultdict(list)
    for _ in range(N):
        s = [[1,0, 0],[0,1, 0],[0,0, 1]][rng.choice([0,1,2],p=[0.7,0.2,0.1])]
        for i in range(length):
            a = random_pi(rng)             # 1) Sample random action
            r, sn = gen_r_sn(s, a, rng)      # 2) Transition to next state and get reward
            if labeled:
                # 3) Store (s, a, r, s') in the trajectory
                trajs[i].append((np.array(s), np.array(a), r))
            else:
                # 3) Store (s, a, s') in the trajectory
                trajs[i].append((np.array(s), np.array(a)))
            s = sn                      # 4) Update current state
    return trajs

def gen_dataset(N1=10,N2=10,H=3,seed=0):
    rng = np.random.RandomState(seed)
    D1 = gen_random_trajs(N=N1, length=H, labeled = True, rng=rng)
    D2 = gen_random_trajs(N=N2, length=H, labeled = False, rng=rng)
    return D1,D2, [[1,0],[0,1]]


if __name__ == '__main__':
    print(gen_dataset())
    pass
