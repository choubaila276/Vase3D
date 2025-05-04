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

with open("data1_with_predictions.txt", "w") as f_out:

vase_points = []
noise_points = []

for i in range(len(X_test)):
    prediction = propagate(X_test[i], Weights)[-1]
    predicted_label = 1 if prediction >= 0.5 else 0


        line = " ".join(map(str, X_test[i])) + f" {predicted_label}\n"
        f_out.write(line)

        if predicted_label == 1:
            vase_points.append(X_test[i])
        else:
            noise_points.append(X_test[i])

vase_points = np.array(vase_points)
noise_points = np.array(noise_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if len(vase_points) > 0:
    ax.scatter(vase_points[:, 0], vase_points[:, 1], vase_points[:, 2],
               c='red', label='Intérieur du vase', s=1)

if len(noise_points) > 0:
    ax.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2],
               c='blue', label='Bruit', s=1)

ax.set_title("Classification des points du vase en 3D")
ax.legend()
plt.show()




        


