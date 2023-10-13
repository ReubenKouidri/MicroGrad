from src.neuron import NeuralNetwork
from src.profiler import profile
import numpy as np
import random

EPOCHS = 500
LR = 0.1


@profile
def train(nn, X, Y, EPOCHS, LR):
    y_final = []
    for epoch in range(EPOCHS):
        ypred = [nn(x) for x in X]
        loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, Y))

        for p in nn.parameters():
            p.grad = 0

        loss.backward()

        for p in nn.parameters():
            p.data -= LR * p.grad

        if epoch == EPOCHS - 1:
            y_final = ypred

    print(f"preds: {y_final}")


if __name__ == '__main__':

    nouts = [4, 4, 1]
    nin = 100
    num_classes = 4
    nn = NeuralNetwork(nin, nouts)  # [3, [4, 4, 1]]

    inputs = [[random.randint(-3, 3) for _ in range(nin)] for c in range(num_classes)]

    # X = [[2.0, 3.0, -1.0],
    #      [3.0, -1.0, 0.5],
    #      [0.5, 1.0, 1.0],
    #      [1.0, 1.0, -1.0]]

    Y = [1.0, -1.0, -1.0, 1.0]

    train(nn, inputs, Y, EPOCHS, LR)
