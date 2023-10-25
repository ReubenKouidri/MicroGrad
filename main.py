from src.neuron import NeuralNetwork
from src.profiler import profile
from src.torch_model import TorchMLP
import random
import torch
import torch.nn as nn
from torch.optim import Adam


random.seed(982375)
torch.random.manual_seed(982375)
EPOCHS = 50
LR = 0.01
TORCH = True


@profile
def train(model, X, Y, EPOCHS, LR):
    if TORCH:
        optimizer = Adam(model.parameters(), LR)
        criterion = nn.MSELoss()
        model.train()
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32).unsqueeze(1)

    for epoch in range(EPOCHS):
        if TORCH:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()

            if epoch == EPOCHS - 1:
                y_final = preds
        else:
            ypred = [model(x) for x in X]
            loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, Y))
            for p in model.parameters():
                p.grad = 0

            loss.backward()
            for p in model.parameters():
                p.data -= LR * p.grad

            if epoch == EPOCHS - 1:
                y_final = ypred

    print(f"preds: {y_final}")


if __name__ == '__main__':
    nouts = [4, 4, 1]
    nin = 3
    num_classes = 4

    if TORCH:
        model = TorchMLP(nin, nouts)
    else:
        model = NeuralNetwork(nin, nouts)

    X = [[random.randint(-3, 3) for _ in range(nin)] for c in range(num_classes)]
    Y = [1.0, -1.0, -1.0, 1.0]
    train(model, X, Y, EPOCHS, LR)
