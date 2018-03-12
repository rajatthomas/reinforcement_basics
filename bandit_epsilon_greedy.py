import numpy as np
import matplotlib.pyplot as plt


class bandit():

    def __init__(self, m):

        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.rand() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1.0 - 1.0/self.N)*self.mean + x/self.N


def bandit_experiment(m1, m2, m3, N, eps):

    bandits = [bandit(m1), bandit(m2), bandit(m3)]

    data = np.empty(N)
    cumulative_mean = []
    for i in range(N):
        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = bandits[j].mean

    cumulative_mean = np.cumsum(data)/(np.arange(N) + 1)
    return cumulative_mean


if __name__ == "__main__":

    m1, m2, m3 = (1, 2, 3)
    N = 10000
    c_p01 = bandit_experiment(m1, m2, m3, N, 0.01)
    #import pdb; pdb.set_trace()
    plt.plot(range(N), c_p01)
    plt.show()
