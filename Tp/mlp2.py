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

