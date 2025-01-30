# Short-Time Fourier Transform implementation.
# First using a short rectangular window (30s),
# then a longer rectangular window (1.5m).
# Shorter window -> best for time resolution
# Longer window -> best for frequency resolution

import scipy.io
from scipy import signal as sig
import numpy as np

Y = scipy.io.loadmat('orthostaticTest.mat')

# Sampling frequency
fs = 1 / np.mean(Y['RR'].squeeze())

# Convert windows length (30s) in number of beats
# because Y['RR'] represents the heartbeats
# Make the window length even because it is better for computing FFT

time = 30
windows_length = time / np.mean(Y['RR'])
if windows_length % 2 != 0:
  windows_length += 1
print(f"Windows length: {windows_length}")

window_pos = np.arange(0, len(Y['RR']) - int(windows_length), windows_length)

matrixFFT = np.zeros((int(windows_length), len(window_pos)))

for pos in window_pos:
  end = int(pos) + int(windows_length)
  #print(f"Position: {int(pos)} and End position: {end}")
  RR_segment = Y['RR'][int(pos):end]
  # Normalize values
  RR_segment_norm = (RR_segment - np.mean(RR_segment)) / np.std(RR_segment)
  win_fft = np.fft.fft(RR_segment_norm)
  matrix_pos = int(pos / windows_length)
  matrixFFT[:, matrix_pos] = win_fft.squeeze()
