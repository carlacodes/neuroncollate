import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping

import h5py
import numpy as np
filepath = 'D:/Electrophysiological Data/F1702_Zola_Nellie/HP_BlockNellie-77/spikeArraysBlockNellie-77BB2andBB3atten0trialsMay-05-2021- 5-56-04-002-PM.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    newarray=np.array(v)
    newarrayremove=newarray[0, :]
    arrays[k] = newarrayremove

filepath2 = 'D:/Electrophysiological Data/F1702_Zola_Nellie/HP_BlockNellie-76/spikeArraysBlockNellie-76BB2andBB3atten0trialsMay-05-2021- 6-20-26-212-PM.mat'
arrays2 = {}
f2 = h5py.File(filepath2)
items2=f2.items()
for k2, v2 in f2.items():
    newarray2=np.array(v2)
    newarrayremove2=newarray2[0, :]
    arrays2[k2] = newarrayremove2
fS=24414.065;
#matplotlib inline
# Trial duration and bin size parameters.
TMIN = 0  # ms
TMAX = int(1.2*fS)   # s*fS
BINSIZE = 100  # ms
NBINS = int((TMAX - TMIN) / BINSIZE)

TMIN2=0
TMAX2=1.2*fS; #I made the maximum trial length 1.2 seconds
# LFP parameters.
LOW_CUTOFF = 10  # Hz
HIGH_CUTOFF = 30  # Hz

# Hyperparameters for shift-only warping model.
SHIFT_SMOOTHNESS_REG = 2000
SHIFT_WARP_REG = 1e-2
MAXLAG = 0.15

# Hyperparameters for linear warping model.
LINEAR_SMOOTHNESS_REG = 1.0
LINEAR_WARP_REG = 0.065

from affinewarp import SpikeData

# Spike times.
S = dict(np.load("umi_spike_data.npz"))
# data = SpikeData(
#     trials=S["trials"],
#     spiketimes=S["spiketimes"],
#     neurons=S["unit_ids"],
#     tmin=TMIN,
#     tmax=TMAX,
# )
# result = arrays["oneDtrialIDarray"];
# result = x[0, :, 0]
adjustedTrial=arrays2["oneDtrialIDarray"]+max(arrays["oneDtrialIDarray"])
combinedTrials=np.concatenate((arrays["oneDtrialIDarray"], adjustedTrial), axis=0)
combinedSpikeTimes=np.concatenate((arrays["oneDspiketimearray"], arrays2["oneDspiketimearray"]),axis=0)
combinedNeuron=np.concatenate((arrays["oneDspikeIDarray"], arrays2["oneDspikeIDarray"]),axis=0)

data2=SpikeData(
    trials=combinedTrials, #arrays["oneDtrialIDarray"],
    spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
    neurons=combinedNeuron, #["oneDspikeIDarray"],
    tmin=TMIN,
    tmax=TMAX,
)
data2.n_neurons=data2.n_neurons.astype(np.int64)
data2.n_trials=data2.n_trials.astype(np.int64)
# Bin and normalize (soft z-score) spike times.
binned = data2.bin_spikes(NBINS)
binned = binned - binned.mean(axis=(0, 1), keepdims=True)
binned = binned / (1e-2 + binned.std(axis=(0, 1), keepdims=True))

# Crop spike times when visualizing rasters.
cropped_data = data2.crop_spiketimes(TMIN, TMAX)

# Load LFP traces (n_trials x n_timebins). Crop traces to [TMIN, TMAX).
L = dict(np.load("umi_lfp_data.npz"))

# Define bandpass filtering function for LFP
from scipy.signal import butter, filtfilt, freqz

def bandpass(x, lowcut, highcut, fs, order=5, axis=-1, kind='butter'):
    """
    Bandpass filters analog time series.

    Parameters
    ----------
    x : ndarray
        Time series data
    lowcut : float
        Defines lower frequency cutoff (e.g. in Hz)
    highcut : float
        Defines upper frequency cutoff (e.g. in Hz)
    fs : float
        Sampling frequency (e.g. in Hz)
    order : int
        Filter order parameter
    kind : str
        Specifies the kind of filter
    axis : int
        Axis along which to bandpass filter data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if kind == "butter":
        b, a = butter(order, [low, high], btype="band")
    else:
        raise ValueError("Filter kind not recognized.")
    return filtfilt(b, a, x, axis=axis)

# Load LFP.
L = dict(np.load("umi_lfp_data.npz"))

# Apply bandpass filter.
lfp = bandpass(L["lfp"], LOW_CUTOFF, HIGH_CUTOFF, L["sample_rate"])

# Crop LFP time base to match spike times.
tidx = (L["lfp_time"] >= TMIN) & (L["lfp_time"] < TMAX)
lfp = lfp[:, tidx]
lfp_time = L["lfp_time"][tidx]

# Z-score LFP.
lfp -= lfp.mean(axis=1, keepdims=True)
lfp /= lfp.std(axis=1, keepdims=True)


# Specify model.
shift_model = ShiftWarping(
    smoothness_reg_scale=SHIFT_SMOOTHNESS_REG,
    warp_reg_scale=SHIFT_WARP_REG,
    maxlag=MAXLAG,
)

# Fit to binned spike times.
shift_model.fit(binned, iterations=50)

# Apply inverse warping functions to data.
shift_aligned_data = shift_model.transform(data2).crop_spiketimes(TMIN, TMAX)

from affinewarp import PiecewiseWarping

# Specify model.
lin_model = PiecewiseWarping(
    n_knots=0,
    smoothness_reg_scale=LINEAR_SMOOTHNESS_REG,
    warp_reg_scale=LINEAR_WARP_REG
)

# Fit to binned spike times.
lin_model.fit(binned, iterations=50)

# Apply inverse warping functions to data.
linear_aligned_data = lin_model.transform(data2).crop_spiketimes(TMIN, TMAX)


plt.plot(shift_model.loss_hist, label="shift")
plt.plot(lin_model.loss_hist, label="linear")
plt.xlabel("Iteration")
plt.ylabel("Normalized Loss")
plt.legend()
plt.tight_layout()
plt.show()

def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

from affinewarp.visualization import rasters
fig, axes=rasters(cropped_data, subplots=(5, 8));
fig.suptitle('Original Data (07/04/2021 Zola) ', fontsize=10, color='1', y='1')

#plt.title('Rasters of Original Data (18/03/2021 Zola) ')
plt.show() #original data

fig, axes=rasters(shift_aligned_data, subplots=(5, 8));
fig.suptitle(' Rasters after Shift Model (07/04/2021 Zola) ', fontsize=10, color='1', y='1')

#plt.title('Rasters after Shift Model (18/03/2021 Zola) ')
plt.show()

fig, axes= rasters(linear_aligned_data, subplots=(5, 8));
fig.suptitle(' Rasters after Linear Model (07/04/2021 Zola) ', fontsize=10, color='1', y='1')

#make_space_above(axes, topmargin=10)

#plt.title('Rasters after Linear Model (18/03/2021 Zola)')
fig.tight_layout()
fig.subplots_adjust(top=10)
plt.show();