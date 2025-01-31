# Short-Time Fourier Transform implementation.
# First using a short rectangular window (30s),
# then a longer rectangular window (1.5m).
# Shorter window -> best for time resolution
# Longer window -> best for frequency resolution

import scipy.io
from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

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

# Frequency and time axis to plot the PSD
freq_axis = [(k*fs)/windows_length for k in range(matrixFFT.shape[0])]
# freq_axis = np.fft.fftfreq(int(windows_length), d=1/fs)
time_axis = Y['timeAxis'][window_pos.astype(int)] + (windows_length / 2) * np.mean(Y['RR'])
'''
for i, pos in enumerate(window_pos):
  end = int(pos) + int(windows_length)

  RR_segment = Y['RR'][int(pos):end]
  # Normalize values
  RR_segment_norm = (RR_segment - np.mean(RR_segment)) / np.std(RR_segment)
  win_fft = np.fft.fft(RR_segment_norm)
  matrixFFT[:, i] = win_fft
'''
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(matrixFFT[:matrixFFT.shape[0]//2,:])**2, aspect="auto", origin="lower", extent=(time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]))
plt.title('Heatmap of the PSD')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Power Spectral Density')
plt.show()
