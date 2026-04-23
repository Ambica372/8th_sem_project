import numpy as np
import time

X_eeg = np.ones((50000, 62, 800), dtype=np.float32)
tr_idx = np.arange(40000)

X_eeg_tr = X_eeg[tr_idx].transpose(0, 2, 1)

eeg_mean = X_eeg_tr.mean(axis=(0,1), keepdims=True)
eeg_std  = X_eeg_tr.std(axis=(0,1), keepdims=True) + 1e-8

start = time.time()
X_eeg_tr -= eeg_mean
X_eeg_tr /= eeg_std
print("Time taken:", time.time() - start)
print("Finished!")
