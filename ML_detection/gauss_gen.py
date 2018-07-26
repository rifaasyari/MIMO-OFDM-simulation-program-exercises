import random
import math

class Gaussgen_obj():

    def __init__(self):
        self.n2 = 0
        self.n2_cached = 0

    def Gaussgen(self, mean, stddev):


        if self.n2_cached == 0:
            x = 0
            y = 0
            r = 0
            while True:
                x = 2.0*random.random() - 1
                y = 2.0*random.random() - 1

                r = x*x + y*y
                if r > 0.0 and r <= 1.0:
                    break
            d = math.sqrt(-2.0*math.log(r)/r)
            n1 = x*d
            self.n2 = y*d
            result = n1*stddev + mean
            self.n2_cached = 1

            return result
        else:
            self.n2_cached = 0
            return self.n2*stddev + mean

    def Gaussgen_matrix(self, mean, stddev, row, column, matrix):
        for i in range(row):
            for j in range(column):
                matrix[i,j] += Gaussgen_obj.Gaussgen(self, mean, stddev) + 1j * Gaussgen_obj.Gaussgen(self, mean, stddev)
