import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pycwt
from pycwt import wavelet

nino3 = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
nino3 = pd.read_table(nino3)
data = nino3.values.squeeze()
N = data.size; print(N)
t0=1871
dt=0.25
units='^{\circ}C'
label='NINO3 SST'
time = np.arange(0, N) * dt + t0

slevel = 0.95                        # Significance level

std = data.std()                      # Standard deviation
std2 = std ** 2                      # Variance
var = (data - data.mean()) / std       # Calculating anomaly and normalizing

dj = 0.25                            # Four sub-octaves per octaves
s0 = -1 #2 * dt                      # Starting scale, here 6 months
J = -1 # 7 / dj                      # Seven powers of two with dj sub-octaves

mother = wavelet.Morlet(6.)          # Morlet mother wavelet with wavenumber=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother)
power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, 0.77,
                        significance_level=slevel, wavelet=mother)
sig95 = (signif * np.ones((N, 1))).transpose()
sig95 = power / sig95                # Where ratio > 1, power is significant

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2 * power.mean(axis=1)
dof = N - scales                     # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, 0.77,
                       significance_level=slevel, dof=dof, wavelet=mother)

# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
f, ax = plt.subplots(figsize=(15,10))
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')
plt.show()
ax.contour(time, np.log2(period), sig95, [-99, 1], colors='k',
           linewidths=2.)
#ax.fill(numpy.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
#        time[:1]-dt, time[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi,
#        [1e-9], period[-1:], period[-1:], [1e-9]])), 'k', alpha='0.3',
#        hatch='x')

ax.plot(time,np.log2(coi), '0.8', lw=4)

ax.plot(time,np.log2(coi), 'w--', lw=3)
#ax.plot(time,numpy.log2(coi), 'k:', lw=3)

ax.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
ax.set_ylabel('Period (years)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
ax.set_yticks(np.log2(Yticks))
ax.set_yticklabels(Yticks)
ax.invert_yaxis()
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], -1);
plt.show()