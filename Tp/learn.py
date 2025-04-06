import numpy as np
from mlp2 import initializer_weights,retroPropagate, propagate

data = np.loadtxt("data.txt")
A = data[:, :-1]
B = data[:, -1:]

Weights = initializer_weights()
Weights = initializer_weights()
for _ in range(10000):
    for i in range(len(A)):
        retroPropagate(A[i], B[i], Weights, alpha=0.1)



for i in range(5):
    output = propagate(A[i], Weights)[-1]
    print(f"Input: {A[i]}, Output: {B[i]}, Expected: {output}")
