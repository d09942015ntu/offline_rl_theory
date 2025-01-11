from collections import defaultdict

import numpy as np

RNG=np.random.RandomState(0)

def gen_r_sn(s,a):
    z = np.matmul(np.array(s)[:,np.newaxis],np.array(a)[np.newaxis,:]).reshape((6,))
    R = np.array([[-0.5, -0.5, -0.2, -0.2, 0.5, 0.5]])
    #10010, 10001, 01010, 01001, 0011, 00101
    P = [
        [1.0, 0.5,   0.0,   0.0,   0.0,   0.5],
        [0.0, 0.5,   1.0,   0.5,   0.0,   0.0],
        [0.0, 0.0,   0.0,   0.5,   1.0,   0.5],
    ]
    si = RNG.choice([0,1,2], p=np.dot(P,z))
    sn=[[1, 0, 0], [0, 1, 0], [0, 0, 1]][si]
    r = np.dot(R,z)[0]
    return r, sn

def random_pi():
    return [[1,0],[0,1]][RNG.choice([0,1])]

def gen_random_trajs(N=10, length=3, labeled = True):
    trajs = defaultdict(list)
    for _ in range(N):
        s = [[1,0, 0],[0,1, 0],[0,0, 1]][RNG.choice([0,1,2],p=[0.7,0.2,0.1])]
        for i in range(length):
            a = random_pi()             # 1) Sample random action
            r, sn = gen_r_sn(s, a)      # 2) Transition to next state and get reward
            if labeled:
                # 3) Store (s, a, r, s') in the trajectory
                trajs[i].append((np.array(s), np.array(a), r))
            else:
                # 3) Store (s, a, s') in the trajectory
                trajs[i].append((np.array(s), np.array(a)))
            s = sn                      # 4) Update current state
    return trajs

def gen_dataset(N1,N2,H):
    D1 = gen_random_trajs(N=N1, length=H, labeled = True)
    D2 = gen_random_trajs(N=N2, length=H, labeled = False)
    return D1,D2


if __name__ == '__main__':
    print(gen_random_trajs())
    print(gen_random_trajs())
    print(gen_random_trajs())
    print(gen_random_trajs())
    print(gen_random_trajs())
    print(gen_random_trajs())
    print(gen_random_trajs())
    print(gen_random_trajs())
