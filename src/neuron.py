from src.value import Value
import random
import abc
# import numpy as np

random.seed(1232453)


class Module(abc.ABC):
    @abc.abstractmethod
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        # out = act(w*xs + b)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # this is a Value object
        out = act.tanh()
        return out
        # act = np.dot(self.w, x) + self.b
        # out = np.tanh(act)
        # return out

    def parameters(self):
        return [self.b] + self.w


class Layer(Module):
    def __init__(self, nin, nout):  # nin = number inputs, nout = num outputs from the layer, i.e. num neurons in layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class NeuralNetwork(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
