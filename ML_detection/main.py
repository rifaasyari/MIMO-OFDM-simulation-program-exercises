import numpy as np
import random
import matplotlib.pyplot as plt
import time
from settings import Settings
import ml_detection as ML



#gaussgen_obj = Gaussgen_obj()



#參數設置
mimo_settings = Settings()











mimo_settings.snr_init()


for i in range(len(mimo_settings.snr)):
    #每個snr跑模擬
    mimo_settings.error = 0
    total = 0
    start_time = time.time()
    for j in range(mimo_settings.N):
        #每個樣本點跑模擬
        mimo_settings.clear()
        No = mimo_settings.Eb / mimo_settings.snr[i]  # 決定雜訊No
        #for m in range(Nt):  # 傳送端一次送出Nt個不同symbol

        mimo_settings.tx_symbol = np.stack(random.sample(mimo_settings.constellation, mimo_settings.Nt)).reshape(mimo_settings.Nr,1) #MIMO的傳送symbol

        mimo_settings.channel_H = np.random.normal(0, 0.5, size = (mimo_settings.Nr, mimo_settings.Nt)) + 1j * np.random.normal(0, 0.5, size = (mimo_settings.Nr, mimo_settings.Nt)) #MIMO的通道矩陣

        mimo_settings.receive_symbol = np.dot(mimo_settings.channel_H, mimo_settings.tx_symbol) + np.random.normal(0, No/2, size = (mimo_settings.Nr, 1)) + 1j * np.random.normal(0, No/2, size = (mimo_settings.Nr, 1))

        #print(mimo_settings.channel_H)


        ML.detection(mimo_settings, 0)

        for m in range(mimo_settings.Nt):
            test_real = abs(mimo_settings.optimal_detection[m, 0].real - mimo_settings.tx_symbol[m, 0].real)
            test_imag = abs(mimo_settings.optimal_detection[m, 0].imag - mimo_settings.tx_symbol[m, 0].imag)
            if  test_real == 2 or test_real == 6:
                mimo_settings.error += 1
            if test_real == 4:
                mimo_settings.error += 2
            if test_imag == 2 or test_imag == 6:
                mimo_settings.error += 1
            if test_imag == 4:
                mimo_settings.error += 2

        mimo_settings.ber[i] = mimo_settings.error / (mimo_settings.K * mimo_settings.Nt * mimo_settings.N)  # 因為一個symbol有K個bit
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    print(mimo_settings.ber)
    print(mimo_settings.optimal_detection - mimo_settings.tx_symbol)


        #gray code










