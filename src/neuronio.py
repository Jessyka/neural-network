import math


class Neuronio:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = []

    def get_output(self, inputs):
        self.inputs = inputs
        self.output = self.calculate_output(self.calculate_total_sum())
        return self.output

    def calculate_output(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_total_sum(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def get_error_from_expected_output(self, target_output):
        return self.calculate_o() * (-(target_output - self.output))

    def calculate_o(self):
        return self.output * (1 - self.output)

    def get_input_by_index(self, index):
        return self.inputs[index]
