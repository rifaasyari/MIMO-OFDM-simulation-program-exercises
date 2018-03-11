import numpy as np
from matplotlib import pyplot as plt
from ray_model import Ray_model
from rice_model import Rice_model


rayleigh_model =  Ray_model(L = 200000)
rice_model = Rice_model(c = 1.2, L = 200000, strat_point = 0, end_point = 4, level = 100)

plt.plot(rayleigh_model.bin_centers, rayleigh_model.histogram, linestyle = "", marker = ".", label = "Histogram of rayleigh_model")
plt.plot(rice_model.bin_centers, rice_model.histogram, linestyle = "", marker = ".", label = "Histogram of rice_model")
plt.legend()
plt.show()