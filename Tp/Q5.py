import numpy as np
from mlp2 import initializer_weights, retroPropagate, propagate
import matplotlib.pyplot as plt

data = np.loadtxt("data.txt")

A = data[:, :-1]
B = data[:, -1:]

# On mélange les indices de manière aléatoire pour éviter tout biais
indices = np.arange(len(A))
np.random.shuffle(indices)

# On réserve 80% des données pour l'entraînement et 20% pour la validation
split_ratio = 0.8
split_point = int(len(A) * split_ratio)

train_indices = indices[:split_point]
val_indices = indices[split_point:]

# Données d'entraînement
X_train = A[train_indices]
y_train = B[train_indices]

# Données de validation
X_val = A[val_indices]
y_val = B[val_indices]
Weights = initializer_weights()


epochs = 20
for epoch in range(epochs):
    for i in range(len(X_train)):
        # Apprentissage par rétropropagation pour chaque point
        retroPropagate(X_train[i], y_train[i], Weights, alpha=0.1)
        r = propagate(X_train[i], Weights)[-1]
        print(f"Input: {X_train[i]}, Output: {r}, Expected: {y_train[i]}")

print("Entraînement terminé avec succès")

vase_points = []
noise_points = []

correct = 0

for i in range(len(X_val)):

    prediction = propagate(X_val[i], Weights)[-1]


    predicted_label = 1 if prediction >= 0.5 else 0
    true_label = int(y_val[i])

    if predicted_label == true_label:
        correct += 1


    if predicted_label == 1:
        vase_points.append(X_val[i])
    else:
        noise_points.append(X_val[i])

accuracy = correct / len(X_val) * 100
print(f"Précision de classification : {accuracy:.2f}%")

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
