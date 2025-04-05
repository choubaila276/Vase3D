
import numpy as np
from mlp import initializer_weights,retroPropagate, propagate  
X = np.array([[0 ,0 ,1], [1 , 1, 1]])
Y = np.array([[1], [0]])

Weights = initializer_weights()

for _ in range(1000):
    for i in range(len(X)):
        retroPropagate(X[i], Y[i], Weights, alpha=0.1)

for i in range(len(X)):
    r = propagate(X[i], Weights)[-1]
    print(f"Input: {X[i]}, Output: {r}, Expected: {Y[i]}")


