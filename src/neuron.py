import math


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # def calculate_pd_error_wrt_total_net_input(self, target_output):
    def calculate_error_with_expected_value(self, target_output):
        return self.calculate_o() * (-(target_output - self.output))

    def calculate_o(self):
        return self.output * (1 - self.output)

    def get_input_by_index(self, index):
        return self.inputs[index]
