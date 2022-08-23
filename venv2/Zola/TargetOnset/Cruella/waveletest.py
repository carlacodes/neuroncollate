from wavelets import WaveletAnalysis
import numpy as np
# given a signal x(t)
x = np.arange(1,1000,1)
# and a sample spacing
dt = 0.1
wa = WaveletAnalysis(x, dt=dt)
# wavelet power spectrum
power = wa.wavelet_power

# scales
scales = wa.scales

# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
T, S = np.meshgrid(t, scales)
ax.contourf(T, S, power, 100)
ax.set_yscale('log')
fig.savefig('test_wavelet_power_spectrum.png')
plt.show()