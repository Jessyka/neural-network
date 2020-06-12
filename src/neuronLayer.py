import random
from neuron import Neuron


class NeuronLayer:
    def __init__(self, num_neurons, bias):
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.get_output(inputs))
        return outputs
