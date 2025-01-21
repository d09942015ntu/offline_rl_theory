import matplotlib.pyplot as plt
import json
import numpy as np

colors = [
    (1, 0, 0),
    (0.7, 0.4, 0),
    (0.7, 0.7, 0),
    (0.4, 0.7, 0),
    (0, 1, 0),
    (0, 0.7, 0.7),
    (0, 0, 1),
]

def linear_regression(n1s,n2s):
    lamda=0
    X = []
    Y = []
    r_exp = []
    for i,n1 in enumerate(n1s):
        jitems= json.load(open(f"results/env_kernel_n1_{n1}.json"))
        rdict = dict( (n2,r) for n2,r in zip(jitems['n2'],jitems['r1']))
        for n2 in n2s:
            if n2 in rdict:
                r = rdict[n2]
                X.append((n1**(-0.5),n2**(-0.5),1))
                Y.append(r)
                r_exp.append(((n1,n2),r))
    r_exp = dict(r_exp)
    X = np.array(X)
    Y = np.array(Y)
    w = np.dot(np.matmul(np.linalg.inv(np.matmul(X.T,X) + lamda*np.eye(X.shape[1])),X.T),Y)
    r_theory = []
    for n1 in n1s:
        for n2 in n2s:
            x = np.array([n1**(-0.5),n2**(-0.5),1])
            r = np.dot(w,x)
            r_theory.append(((n1,n2),r))
    r_theory = dict(r_theory)
    return r_exp, r_theory


def visualize_distribution_exp():
    plt.clf()
    plt.figure(figsize=(10, 6))
    n1s = [10,20,50,100,200,500]
    for i,n1 in enumerate(n1s):
        jitems= json.load(open(f"results/env_kernel_n1_{n1}.json"))
        plt.plot(jitems['n2'], jitems['r1'], label=f'N_1={n1}', color=colors[i])
        #if n == n1s[0]:
        #    plt.axhline(y=jitems['rrand'], color='r', linestyle='--', label=f'random = {r_rand}')
    plt.xlabel('$N_2$')
    plt.ylabel('$V_1^{\pi}(s)$')
    plt.title('Plot of $V_1^{\pi}(s)$ vs $N_2$ and $N_1$')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/out_exp.png")

def regression_theory():
    n1s = [10,20,50,100]
    n2s = [10,20,50,100,200,500]
    r_exp, r_theory = linear_regression(n1s,n2s)
    visualize_distribution_theory(n1s,n2s,r_exp,"exp")
    visualize_distribution_theory(n1s,n2s,r_theory,"theory")

def visualize_distribution_theory(n1s,n2s,r_vals, fname):
    plt.clf()
    plt.figure(figsize=(10, 6))
    for i,n1 in enumerate(n1s):
        r1e = []
        n2s_plot = []
        for n2 in n2s:
            print(n1, n2)
            if (n1,n2) in r_vals.keys():
                r1e.append(r_vals[(n1,n2)])
                n2s_plot.append(n2)
        plt.plot(n2s_plot, r1e, label=f'$N_1={n1}$', color=colors[i])
        #if n == n1s[0]:
        #    plt.axhline(y=jitems['rrand'], color='r', linestyle='--', label=f'random = {r_rand}')
    plt.xlabel('$N_2$')
    plt.ylabel('$V_1^{\pi}(s)$')
    plt.title('Plot of $V_1^{\pi}(s)$ vs $N_2$ and $N_1$')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/out_{fname}.png")


if __name__ == '__main__':
    regression_theory()