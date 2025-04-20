import numpy as np
from mlp2 import initializer_weights, retroPropagate, propagate

data = np.loadtxt("data.txt")
A = data[:, :-1]
B = data[:, -1:]

Weights = initializer_weights()

for i in range(len(A)):
    for _ in range(2500):
        retroPropagate(A[i], B[i], Weights, alpha=0.1)

    r = propagate(A[i], Weights)[-1]
    print(f"Input: {A[i]}, Output: {r}, Expected: {B[i]}")
