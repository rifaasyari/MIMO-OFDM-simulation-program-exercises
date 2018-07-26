import numpy as np



def detection(mimo_settings, current):
    if current != mimo_settings.Nr:
        for i in range(mimo_settings.len):
            mimo_settings.detect.itemset(current,mimo_settings.constellation[i])
            detection(mimo_settings, current + 1)
    else:
        mimo_settings.detect_y = np.dot(mimo_settings.channel_H, mimo_settings.detect)
        #distance = np.linalg.norm(mimo_settings.receive_symbol - mimo_settings.detect_y) ** 2
        distance = np.linalg.norm(mimo_settings.receive_symbol - mimo_settings.detect_y)

        if distance < mimo_settings.min_distance:
            mimo_settings.min_distance = distance
            mimo_settings.optimal_detection = mimo_settings.detect.copy()
