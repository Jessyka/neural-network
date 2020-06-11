import numpy as np
import csv
import random
from neuralNetwork import NeuralNetwork


def init():
    # Leitura do data set
    data = leitura_csv('data/XOR_Training.csv')

    # Fase de Treinamento
    neuralNetwork = NeuralNetwork(5, len(data[0][0]), len(data[0][1]))
    for i in range(10000):
        print('Treinamento: ', i)
        training_inputs, training_outputs = random.choice(data)
        neuralNetwork.treinamento(training_inputs, training_outputs)


    print('Treinamento concluido')
    print('Entradas: [0, 0.99] = ', neuralNetwork.feed_forward([0, 0.99]))


def leitura_csv(fileName):
    print(f'Leitura do arquivo csv {fileName}')
    data = []
    with open(fileName, 'r') as arquivo_csv:
        leitor = csv.reader(arquivo_csv, delimiter=',')
        for coluna in leitor:
            data.append([[float(coluna[0]), float(coluna[1])], [float(coluna[2])]])
    return np.random.permutation(data)


init()

