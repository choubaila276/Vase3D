import numpy as np
from mlp2 import initializer_weights, retroPropagate, propagate
import matplotlib.pyplot as plt

train_data = np.loadtxt("data.txt")

X_train = train_data[:, :-1]   
y_train = train_data[:, -1:]
indices = np.arange(len(X_train))  
np.random.shuffle(indices)        
X_train = X_train[indices]
y_train = y_train[indices]

X_test = np.loadtxt("data1.txt")   
Weights = initializer_weights()
epochs = 20
for epoch in range(epochs):
    for i in range(len(X_train)):
        retroPropagate(X_train[i], y_train[i], Weights, alpha=0.1)

vase_points = []
noise_points = []

for i in range(len(X_test)):
    prediction = propagate(X_test[i], Weights)[-1]
    print(f"Input: {X_test[i]}, Output: {prediction }")

    predicted_label = 1 if prediction >= 0.5 else 0

    if predicted_label == 1:
        vase_points.append(X_test[i])
    else:
        noise_points.append(X_test[i])



        


