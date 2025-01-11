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

def gen_random_traj(init=[1,0,0], length=10, labeled = True):
    traj = []
    s = init
    for _ in range(length):
        a = random_pi()             # 1) Sample random action
        r, sn = gen_r_sn(s, a)      # 2) Transition to next state and get reward
        if labeled:
            # 3) Store (s, a, r, s') in the trajectory
            traj.append((s, a, r))
        else:
            # 3) Store (s, a, s') in the trajectory
            traj.append((s, a))
        s = sn                      # 4) Update current state
    return traj

if __name__ == '__main__':
    print(gen_r_sn([1,0,0],[1,0]))
    print(gen_r_sn([1, 0, 0], [0, 1]))
    print(gen_r_sn([0, 1, 0], [1, 0]))
    print(gen_r_sn([0, 1, 0], [0, 1]))
    print(gen_r_sn([0, 0, 1], [1, 0]))
    print(gen_r_sn([0, 0, 1], [0, 1]))
