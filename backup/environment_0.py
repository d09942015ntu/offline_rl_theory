import numpy as np

def gen_dataset(N1,N2,H):
    D1 = [[] for _ in range(H)]
    # E.g., for h=0, have a few (s,a,r)
    D1[0] = [
        (np.array([0.1, 0.2]), np.array([0.0]), 0.5),
        (np.array([0.4, 0.1]), np.array([1.0]), 0.7),
    ]
    D1[1] = [
        (np.array([0.9, 0.8]), np.array([0.0]), 0.3),
    ]
    D1[2] = [
        (np.array([0.2, 0.3]), np.array([1.0]), 0.9),
    ]

    # D2[h] is a list of (s_h, a_h) with no r
    D2 = [[] for _ in range(H)]
    D2[0] = [
        (np.array([0.7, 0.2]), np.array([0.0])),
    ]
    D2[1] = []
    D2[2] = [
        (np.array([0.8, 0.8]), np.array([1.0])),
    ]
