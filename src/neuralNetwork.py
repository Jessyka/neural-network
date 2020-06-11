import random
from neuronLayer import NeuronLayer


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_cam_oculta, total_inputs, total_outputs, cam_oculta_bias=None, cam_saida_bias=None):
        self.total_inputs = total_inputs

        self.hidden_layer = NeuronLayer(num_cam_oculta, cam_oculta_bias)
        self.output_layer = NeuronLayer(total_outputs, cam_saida_bias)

        # Calculando pesos das camadas
        self.init_pesos_para_camada_oculta()
        self.init_pesos_para_camada_saida()

    def init_pesos_para_camada_oculta(self):
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.total_inputs):
                self.hidden_layer.neurons[h].weights.append(random.random())

    def init_pesos_para_camada_saida(self):
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                self.output_layer.neurons[o].weights.append(random.random())

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def treinamento(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        errors_camada_saida = self.encontre_error_camada_saida(training_outputs)
        errors_camada_oculta = self.encontre_error_camada_oculta(errors_camada_saida)

        # Atualizacao de pesos camada saida
        for index in range(len(self.output_layer.neurons)):
            for w_index in range(len(self.output_layer.neurons[index].weights)):
                error_weight = errors_camada_saida[index] * self.output_layer.neurons[index].get_input_by_index(w_index)
                self.output_layer.neurons[index].weights[w_index] -= self.LEARNING_RATE * error_weight

        # Atualizacao de pesos camada oculta
        for index in range(len(self.hidden_layer.neurons)):
            for w_index in range(len(self.hidden_layer.neurons[index].weights)):
                error_weight = errors_camada_oculta[index] * self.hidden_layer.neurons[index].get_input_by_index(w_index)
                self.hidden_layer.neurons[index].weights[w_index] -= self.LEARNING_RATE * error_weight

    def encontre_error_camada_saida(self, training_outputs):
        erros_camada_saida = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            erros_camada_saida[o] = self.output_layer.neurons[o].calculate_error_with_expected_value(training_outputs[o])

        return erros_camada_saida

    def encontre_error_camada_oculta(self, errors_camada_saida):
        errors_camada_escondida = [0] * len(self.hidden_layer.neurons)

        for index in range(len(self.hidden_layer.neurons)):
            sum_esperado_e_peso_camada_oculta = 0
            for cam_saida_index in range(len(self.output_layer.neurons)):
                sum_esperado_e_peso_camada_oculta += errors_camada_saida[cam_saida_index] * \
                                                    self.output_layer.neurons[cam_saida_index].weights[index]

            errors_camada_escondida[index] = sum_esperado_e_peso_camada_oculta * self.hidden_layer.neurons[index].calculate_o()
        return errors_camada_escondida
