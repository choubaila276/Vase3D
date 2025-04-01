import numpy as np
import random
layers_card = [3,6,3,1]
def initializer_weights():
    Weights = []
    precedent_card = 0

    for card in layers_card:
        if precedent_card == 0:
            precedent_card = card
        else:
            W = []
            for i in range(precedent_card + 1):
                W.append([np.random.uniform(-1, 1) for j in range(card)])
            precedent_card = card
            Weights.append(np.array(W))

    return Weights


def propagate(X,W):
	def f(X): return 1./(1+np.exp(-X))
	P = [np.array(X)]  
	for w in W:
		X = np.append(1,X)
		X = f(np.dot(X,w))
		P += [X]
	return P

def retroPropagate(X, Y, Weights, alpha=0.05):
    S = propagate(X, Weights)
    Deltas = [np.zeros(card) for card in layers_card]

    delta_index = len(Deltas) - 1
    Deltas[delta_index] = S[delta_index] * (1. - S[delta_index]) * (Y - S[delta_index])

    for delta_index in range(len(Deltas) - 2, -1, -1):
        S_mul = S[delta_index]
        W_mul = Weights[delta_index][1:] 
        Deltas[delta_index] = S_mul * (1. - S_mul) * np.dot(Deltas[delta_index + 1], W_mul.T)

    for Weights_index in range(len(Weights)):
        Deltas_mul = Deltas[Weights_index + 1]
        S_mul = np.append(1, S[Weights_index])

        for m in range(len(Weights[Weights_index])):
            Weights[Weights_index][m] += alpha * Deltas_mul * S_mul[m]

    return Weights
