import numpy as np
import random


class Settings():
    def __init__(self):
        self.snr_db = [0] * 13
        self.snr = [0] * len(self.snr_db)
        self.ber = [0] * len(self.snr_db)
        self.N = 1  # 執行N次來找錯誤率
        self.Nt = 8 # 傳送端天線數
        self.Nr = 8  # 接收端天線數
        self.Eb = 0
        self.error = 0

        # 這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
        self.channel_H = np.zeros(shape=(self.Nr, self.Nt), dtype=complex)
        self.tx_symbol = np.zeros(shape=(self.Nt, 1), dtype=complex)
        self.receive_symbol = np.zeros(shape=(self.Nr, 1), dtype=complex)
        self.detect_y = np.zeros(shape=(self.Nr, 1), dtype=complex)
        self.detect = np.zeros(shape=(self.Nt, 1), dtype=complex)
        self.optimal_detection = np.zeros(shape=(self.Nt, 1), dtype=complex)
        self.min_distance = 99999
        self.num = 0

        # 這裡以16-QAM作為模擬
        self.constellation = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, -1 - 1j, -1 - 3j, -3 - 1j,
                         -3 - 3j, 1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j]
        self.len = len(self.constellation)
        self.constellation_new = [-3, -1, 1, 3]
        # 實部、虛部值域皆為{ -3, -1, 1, 3}
        self.K = 0
        self.N0 = 0

    def snr_init(self):
        for i in range(len(self.snr)):
            self.snr_db[i] = 0.5 + 2.5 * i
            self.snr[i] = np.power(10, self.snr_db[i] / 10)
        self.K = int(np.log2(self.len))
        energy = np.linalg.norm(self.constellation) ** 2
        Es = energy / self.len
        self.Eb = Es / self.K
        #print(energy)

    def tx_symbol_gen(self):
        self.tx_symbol = np.stack(random.sample(self.constellation, self.Nt)).reshape(self.Nt, 1)

    def channel_H_gen(self):
        self.channel_H = np.random.normal(0, 1 / np.sqrt(2),
                                          size = (self.Nr, self.Nt)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size = (self.Nr, self.Nt))

    def receive_symbol_gen(self, No):
        self.receive_symbol = np.dot(self.channel_H, self.tx_symbol) + \
                              np.random.normal(0, np.sqrt(No/2), size = (self.Nr, 1)) + 1j * np.random.normal(0, np.sqrt(No/2), size = (self.Nr, 1))
