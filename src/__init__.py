import numpy as np
import csv
from perceptronSimples import PerceptronSimples

def init():
    # Leitura do data set
    data = leitura_csv('data/portaAND_Original.csv')

    # Separar base de treinamento e base de teste
    treinamento_size = int(200 * 0.8)
    baseTreinamento = data[0: treinamento_size][:]
    baseTeste = data[treinamento_size: 200][:]

    # Realizar treinamento
    perceptronSimples = PerceptronSimples()
    perceptronSimples.treinamento(baseTreinamento, treinamento_size)

    # Realizar test
    perceptronSimples.test(baseTeste, (200 - treinamento_size))

def leitura_csv(fileName):
    print(f'Leitura do arquivo csv {fileName}')
    data = []
    with open(fileName, 'r') as arquivo_csv:
        leitor = csv.reader(arquivo_csv, delimiter=',')
        for coluna in leitor:
            data.append([float(coluna[0]), float(coluna[1]), float(coluna[2])])
    return np.random.permutation(data)

init()

