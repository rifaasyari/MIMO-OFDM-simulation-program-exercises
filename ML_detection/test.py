import numpy as np


xx = np.random.normal(0, 1 / np.sqrt(2), size = (2, 2)) + 1J * np.random.normal(0, 1 / np.sqrt(2), size = (2, 2))
print(np.linalg.norm(xx))
H = [[0j]*2 for i in range(2)]
H = np.matrix(H)
for m in range(2):
    for n in range(2):
        H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()
print(np.linalg.norm(H))