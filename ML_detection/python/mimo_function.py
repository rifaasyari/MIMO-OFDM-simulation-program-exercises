import numpy as np

def clear_error(mimo_settings):
    mimo_settings.error = 0

def check_and_sum_bit_error(mimo_settings, i):
    for m in range(mimo_settings.Nt):
        test_real = abs(mimo_settings.optimal_detection[m, 0].real - mimo_settings.tx_symbol[m, 0].real)
        test_imag = abs(mimo_settings.optimal_detection[m, 0].imag - mimo_settings.tx_symbol[m, 0].imag)
        if test_real == 2 or test_real == 6:
            mimo_settings.error += 1
        if test_real == 4:
            mimo_settings.error += 2
        if test_imag == 2 or test_imag == 6:
            mimo_settings.error += 1
        if test_imag == 4:
            mimo_settings.error += 2
    mimo_settings.ber[i] = mimo_settings.error / (mimo_settings.K * mimo_settings.Nt * mimo_settings.N)

def clear_and_prepare_settings(mimo_settings, i):
    mimo_settings.channel_H = np.zeros(shape=(mimo_settings.Nr, mimo_settings.Nt), dtype=complex)
    mimo_settings.tx_symbol = np.zeros(shape=(mimo_settings.Nt, 1), dtype=complex)
    mimo_settings.receive_symbol = np.zeros(shape=(mimo_settings.Nr, 1), dtype=complex)
    mimo_settings.detect_y = np.zeros(shape=(mimo_settings.Nr, 1), dtype=complex)
    mimo_settings.detect = np.zeros(shape=(mimo_settings.Nt, 1), dtype=complex)
    mimo_settings.optimal_detection = np.zeros(shape=(mimo_settings.Nt, 1), dtype=complex)
    mimo_settings.min_distance = 99999
    mimo_settings.num = 0
    mimo_settings.N0 = mimo_settings.Eb / mimo_settings.snr[i]


def ML_detection(mimo_settings, current):
    if current != mimo_settings.Nr:
        for i in range(mimo_settings.len):
            mimo_settings.detect.itemset(current,mimo_settings.constellation[i])
            ML_detection(mimo_settings, current + 1)
    else:
        mimo_settings.detect_y = np.dot(mimo_settings.channel_H, mimo_settings.detect)
        #distance = np.linalg.norm(mimo_settings.receive_symbol - mimo_settings.detect_y) ** 2
        distance = np.linalg.norm(mimo_settings.receive_symbol - mimo_settings.detect_y)

        if distance < mimo_settings.min_distance:
            mimo_settings.min_distance = distance
            mimo_settings.optimal_detection = mimo_settings.detect.copy()