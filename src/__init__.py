import numpy as np
import csv
import random
from neuralNetwork import NeuralNetwork


def init():
    # Leitura do data set
    print('Iniciando leitura base de dados...')
    data = leitura_csv('data/XOR_Training.csv')
    training_data = get_training_data(data)
    test_data = get_test_data(data)

    # Fase de Treinamento
    print('Iniciando Fase de Treinamento...')
    neuralNetwork = NeuralNetwork(5, len(training_data[0][0]), len(training_data[0][1]))
    for i in range(1000):
        training_inputs, training_outputs = random.choice(training_data)
        neuralNetwork.training(training_inputs, training_outputs)
    print('Treinamento concluido')

    #Fase de Teste
    print('Iniciando fase de teste:')
    error_sum = 0
    for i in range(len(test_data)):
        test_inputs, test_outputs = test_data[i][:]
        calculated_output = neuralNetwork.feed_forward(test_inputs)
        if not is_valid_output(test_outputs, calculated_output):
            error_sum += 1

    print('Total de itens na base de teste: ', len(test_data))
    print('Total de acertos: ', len(test_data) - error_sum)
    print('Erros na fase de teste: ', error_sum)

def leitura_csv(fileName):
    print(f'Leitura do arquivo csv {fileName}')
    data = []
    with open(fileName, 'r') as arquivo_csv:
        leitor = csv.reader(arquivo_csv, delimiter=',')
        for coluna in leitor:
            data.append([[float(coluna[0]), float(coluna[1])], [float(coluna[2])]])
    return np.random.permutation(data)

def get_training_data(data):
    #Treinamento com 80% da base
    treinamento_size = int(len(data) * 0.8)
    return data[0: treinamento_size][:]

def get_test_data(data):
    #Teste com 20% da base
    treinamento_size = int(len(data) * 0.8)
    return data[treinamento_size: 200][:]

def is_valid_output(expectedValue, output):
    for index_output in range(len(expectedValue)):
        if int(expectedValue[index_output]) != int(output[index_output]):
            return False
    return True

init()

