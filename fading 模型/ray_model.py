import numpy as np


class Ray_model():

    def __init__(self, L, strat_point = 0, end_point = 5, level = 60):

        # Sample from a rayleigh distribution using numpy's random number generator
        self.samples = np.random.rayleigh(size = L)
        self.bins = np.linspace(strat_point, end_point, level)
        self.histogram, self.bins = np.histogram(self.samples, bins = self.bins, normed=True)
        self.bin_centers = 0.5*(self.bins[1:] + self.bins[:-1])