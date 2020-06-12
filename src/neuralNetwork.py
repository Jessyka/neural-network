import random
from neuronLayer import NeuronLayer


class NeuralNetwork:

    def __init__(self, num_hidden_layer, total_inputs, total_outputs, bias_hidden_layer=None, bias_output_layer=None,
                 learning_rate=None):

        self.total_inputs = total_inputs
        self.learning_rate = learning_rate if learning_rate else 0.5

        self.hidden_layer = NeuronLayer(num_hidden_layer, bias_hidden_layer)
        self.output_layer = NeuronLayer(total_outputs, bias_output_layer)

        self.init_weight_hidden_layer()
        self.init_output_hidden_layer()

    def init_weight_hidden_layer(self):
        for index_hidden_layer in range(len(self.hidden_layer.neurons)):
            for index_total_inputs in range(self.total_inputs):
                self.hidden_layer.neurons[index_hidden_layer].weights.append(random.random())

    def init_output_hidden_layer(self):
        for index_output_layer in range(len(self.output_layer.neurons)):
            for index_hidden_layer in range(len(self.hidden_layer.neurons)):
                self.output_layer.neurons[index_output_layer].weights.append(random.random())

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def training(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        error_output_layer = self.find_error_output_layer(training_outputs)
        error_hidden_layer = self.find_error_hidden_layer(error_output_layer)

        self.update_output_layer_weights(error_output_layer)
        self.update_hidden_layer_weights(error_hidden_layer)

    def find_error_output_layer(self, training_outputs):
        error_output_layer = [0] * len(self.output_layer.neurons)

        for index_output_layer in range(len(self.output_layer.neurons)):
            error_output_layer[index_output_layer] = self.output_layer.neurons[index_output_layer].get_error_from_expected_output(training_outputs[index_output_layer])

        return error_output_layer

    def find_error_hidden_layer(self, errors_output_layer):
        errors_hidden_layer = [0] * len(self.hidden_layer.neurons)

        for index in range(len(self.hidden_layer.neurons)):
            sum_expected_weight = 0
            for index_output_layer in range(len(self.output_layer.neurons)):
                sum_expected_weight += errors_output_layer[index_output_layer] * self.output_layer.neurons[index_output_layer].weights[index]
            errors_hidden_layer[index] = sum_expected_weight * self.hidden_layer.neurons[index].calculate_o()

        return errors_hidden_layer

    def update_hidden_layer_weights(self, error_output_layer):
        for index in range(len(self.hidden_layer.neurons)):
            for w_index in range(len(self.hidden_layer.neurons[index].weights)):
                error_weight = error_output_layer[index] * self.hidden_layer.neurons[index].get_input_by_index(w_index)
                self.hidden_layer.neurons[index].weights[w_index] -= self.learning_rate * error_weight

    def update_output_layer_weights(self, error_hidden_layer):
        for index in range(len(self.output_layer.neurons)):
            for w_index in range(len(self.output_layer.neurons[index].weights)):
                error_weight = error_hidden_layer[index] * self.output_layer.neurons[index].get_input_by_index(w_index)
                self.output_layer.neurons[index].weights[w_index] -= self.learning_rate * error_weight
