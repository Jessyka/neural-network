from random import uniform


class PerceptronSimples:

    def __init__(self):
        self._x = -1
        self._n = 0.1
        self._totalEpocas = 1000
        self._w = [uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)]

    @property
    def n(self):
        return self._n

    @property
    def w(self):
        return self._w

    def treinamento(self, baseTreinamento, size):
        print('Fase: Treinamento')
        print('--> Iniciando Treinamento')

        for x in range(self._totalEpocas):
            for i in range(size):
                error = 1
                while error == 1:
                    u = self.w[0] * self._x + baseTreinamento[i][0] * self.w[1] + baseTreinamento[i][1] * self.w[2]
                    y = 1 if u > 0 else 0
                    d = 1 if baseTreinamento[i][2] == 1 else 0

                    error = d - y
                    self.updateVetorW(error, baseTreinamento[i][:])

        print('--> Fim Treinamento')

    def updateVetorW(self, error, x):
        if error != 0:
            self._w = [self._w[0] + self._n * error * self._x,
                       self._w[1] + self._n * error * x[0],
                       self._w[2] + self._n * error * x[1]]

    def test(self, baseTeste, test_size):
        print('Fase: Teste')
        acerto = 0

        for i in range(test_size):
            u = self.w[0] * self._x + baseTeste[i][0] * self.w[1] + baseTeste[i][1] * self.w[2]
            y = 1 if u > 0 else 0
            d = 1 if baseTeste[i][2] == 1 else 0

            if d == y:
                acerto += 1

        print(f'--> Acertos: {acerto} Erros: {(test_size - acerto)}')
