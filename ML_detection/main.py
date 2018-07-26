#!/usr/bin/env python
import matplotlib.pyplot as plt
import time
from settings import *
import mimo_function as mf


mimo_settings = Settings()
mimo_settings.snr_init()

for i in range(len(mimo_settings.snr)):

    mf.clear_error(mimo_settings)
    start_time = time.time()
    for j in range(mimo_settings.N):
        mf.clear_and_prepare_settings(mimo_settings, i)
        mimo_settings.tx_symbol_gen()
        mimo_settings.channel_H_gen()
        mimo_settings.receive_symbol_gen(mimo_settings.N0)
        mf.ML_detection(mimo_settings, 0)
        mf.check_and_sum_bit_error(mimo_settings, i)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))

plt.semilogy(mimo_settings.snr_db, mimo_settings.ber, marker='o', label='ML detection for 16QAM (Nt=1 , Nr=1 )')
plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.legend()
plt.grid(True, which='both')
plt.show()
