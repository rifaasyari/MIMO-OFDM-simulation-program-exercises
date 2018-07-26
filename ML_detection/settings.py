import numpy as np


class Settings():

    def __init__(self):

        self.snr_db = [0] * 13
        self.snr = [0] * len(self.snr_db)
        self.ber = [0] * len(self.snr_db)
        self.N = 1000  # 執行N次來找錯誤率
        self.Nt = 2  # 傳送端天線數
        self.Nr = 2  # 接收端天線數
        self.Eb = 0
        self.error = 0


        # 這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
        self.channel_H = np.zeros(shape=(self.Nr,self.Nt),dtype=complex)
        self.tx_symbol = np.zeros(shape=(self.Nt,1),dtype=complex)
        self.receive_symbol = np.zeros(shape=(self.Nr,1),dtype=complex)
        self.detect_y = np.zeros(shape=(self.Nr,1),dtype=complex)
        self.detect = np.zeros(shape=(self.Nt,1),dtype=complex)
        self.optimal_detection = np.zeros(shape=(self.Nt,1),dtype=complex)
        self.min_distance = 99999

        self.num = 0

        # 這裡以16-QAM作為模擬
        self.constellation = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, -1 - 1j, -1 - 3j, -3 - 1j,
                         -3 - 3j, 1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j]
        self.len = len(self.constellation)
        self.constellation_new = [-3, -1, 1, 3]
        # 實部、虛部值域皆為{ -3, -1, 1, 3}

        self.K = 0

    def clear(self):
        self.channel_H = np.zeros(shape=(self.Nr, self.Nt), dtype=complex)
        self.tx_symbol = np.zeros(shape=(self.Nt, 1), dtype=complex)
        self.receive_symbol = np.zeros(shape=(self.Nr, 1), dtype=complex)
        self.detect_y = np.zeros(shape=(self.Nr, 1), dtype=complex)
        self.detect = np.zeros(shape=(self.Nt,1),dtype=complex)
        self.optimal_detection = np.zeros(shape=(self.Nt, 1), dtype=complex)
        self.min_distance = 99999
        self.num = 0


    def snr_init(self):
        for i in range(len(self.snr)):
            self.snr_db[i] = 3 * i
            self.snr[i] = np.power(10, self.snr_db[i] / 10)

        self.K = int(np.log2(self.len))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        energy = np.linalg.norm(self.constellation) ** 2
        Es = energy / self.len # 平均一個symbol有Es的能量

        self.Eb = Es / self.K  # 平均一個bit有Eb能量
