import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping
import os
import h5py
import numpy as np


#user_input = input('What is the name of your directory')
f={}
blockData={}
blocksOfInterest=[100,101,102,103,104,105,106,108,110]
for i in blocksOfInterest:
    user_input = 'D:/Electrophysiological Data/F1702_Zola_Nellie/HP_BlockNellie-'+str(i)+'/comdist/nopitchshift/orderingbyLRtime/nomisses2s'
    directory = os.listdir(user_input)

    searchstring = 'Arrays'#input('What word are you trying to find?')

    for fname in directory:
        if searchstring in fname:
            # Full path
            f[i] = h5py.File(user_input + os.sep + fname)
            items = f[i].items()
            arrays = {}
            for k3, v3 in f[i].items():
                newarray3 = np.array(v3)
                newarrayremove3 = newarray3[0, :]
                arrays[k3] = newarrayremove3
            blockData[i]=arrays



            f[i].close()


TMIN = 0*1000  # s


TMIN2=0
TMAX2=1.2; #I made the maximum trial length 1.2 seconds
# LFP parameters.
LOW_CUTOFF = 10  # Hz
HIGH_CUTOFF = 30  # Hz

# Hyperparameters for shift-only warping model.
SHIFT_SMOOTHNESS_REG = 0.5
SHIFT_WARP_REG = 1e-2
MAXLAG = 0.15

# Hyperparameters for linear warping model.
LINEAR_SMOOTHNESS_REG = 1.0
LINEAR_WARP_REG = 0.065

from affinewarp import SpikeData

# Spike times.
S = dict(np.load("umi_spike_data.npz"))

adjustedTrial={}
for i2 in range(len(blocksOfInterest)-1):
    if i2==0:
        adjustedTrial[i2]=blockData[blocksOfInterest[i2+1]]["oneDtrialIDarray"]+max(blockData[blocksOfInterest[i2]]["oneDtrialIDarray"])
    else:
        adjustedTrial[i2]=blockData[blocksOfInterest[i2+1]]["oneDtrialIDarray"]+max(adjustedTrial[i2-1])

combinedTrialsAdjusted=np.concatenate([v for k,v in sorted(adjustedTrial.items())], 0)
firsttrialarray=blockData[blocksOfInterest[0]]["oneDtrialIDarray"]
combinedTrials=np.append(firsttrialarray, combinedTrialsAdjusted)
for i in range(len(combinedTrials)):
    combinedTrials[i] -= 1

combinedSpikeTimes=np.array([]); #declare empty numpy array
combinedNeuron=np.array([])
combinedLickReleaseTimes=np.array([])

for i3 in range(len(blockData)):
    selectedSpikeTimes=blockData[blocksOfInterest[i3]]["oneDspiketimearray"]
    selectedNeuronIDs=blockData[blocksOfInterest[i3]]["oneDspikeIDarray"]
    selectedLickReleaseIDs=blockData[blocksOfInterest[i3]]["oneDlickReleaseArray"]
    combinedSpikeTimes=np.append(combinedSpikeTimes,selectedSpikeTimes)
    combinedNeuron=np.append(combinedNeuron, selectedNeuronIDs)
    combinedLickReleaseTimes=np.append(combinedLickReleaseTimes,selectedLickReleaseIDs)
TMAX = 0.8*1000#np.mean(combinedLickReleaseTimes)# ms
BINSIZE = 0.01*1000  # 10 ms
NBINS = int((TMAX - TMIN) / BINSIZE)
#combinedSpikeTimes=np.concatenate([v for k,v in sorted(blockData.items())], key='oneDspiketimearray',  axis=0)

#adjustedTrial=arrays2["oneDtrialIDarray"]+max(arrays["oneDtrialIDarray"])
#adjustedTrial2=arrays3["oneDtrialIDarray"]+max(adjustedTrial)
# combinedTrials=np.concatenate((arrays["oneDtrialIDarray"], adjustedTrial, adjustedTrial2), axis=0)
# combinedSpikeTimes=np.concatenate((arrays["oneDspiketimearray"], arrays2["oneDspiketimearray"], arrays3["oneDspiketimearray"]),axis=0)
# combinedNeuron=np.concatenate((arrays["oneDspikeIDarray"], arrays2["oneDspikeIDarray"], arrays3["oneDspikeIDarray"]),axis=0)

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

trialrows= np.array([])
maxcombinedTrials=int(max(combinedTrials))
for i in range(maxcombinedTrials+1):
    #trialrows.append((i)+1)
    trialrows=np.append(trialrows,float(i))

trialalignment=np.concatenate((combinedLickReleaseTimes.reshape(-1,1),trialrows.reshape(-1,1)),axis=1)
#indTrial=np.argsort(trialalignment[:,0])
sorted_array = trialalignment[np.argsort(trialalignment[:, 0])]
sorted_array_trial=sorted_array[:,1]
sorted_array_trial=(sorted_array_trial).astype(np.int)
#t3 = np.concatenate((t1.reshape(-1,1),t2.reshape(-1,1),axis=1)

#data3=data2.select_trials([1,2,3,4,5])
#data4=data3.reorder_trials([0,1,3,2,4])

data22=data2.reorder_trials(sorted_array_trial)
# Bin and normalize (soft z-score) spike times.
binnedLR = data2.bin_spikes(NBINS)
binnedLR = binnedLR - binnedLR.mean(axis=(0, 1), keepdims=True)
binnedLR = binnedLR / (1e-2 + binnedLR.std(axis=(0, 1), keepdims=True))


cropped_data2 = data22.crop_spiketimes(TMIN, TMAX)
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
#trialrows=np.array([ i in (max((combinedTrials)))])

lin_model.fit(binnedLR, iterations=50)

# Apply inverse warping functions to data.
linear_aligned_dataLR = lin_model.transform(data22).crop_spiketimes(TMIN, TMAX)

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
import numpy as np
import matplotlib.pyplot as plt

##adding yticks with the actual lick release time in ms relative to the start trial lick

from visualization1006 import rasters
fig, axes=rasters(cropped_data, sorted_array,(5, 8), style='white');
fig.suptitle('Original Data Aligned to Distractor (all lick releases 07/06/2021 Zola) ', fontsize=10, color='0', y='1')

plt.show() #original data

fig, axes=rasters(cropped_data2,sorted_array, subplots=(5, 8), style='white');
fig.suptitle('Original Data Ordered, Aligned to Distractor (CORRECT non Pitch Shift lick releases 07/06/2021 Zola) ', fontsize=10, color='0', y='1')

plt.show() #original data

fig, axes=rasters(shift_aligned_data, sorted_array, subplots=(5, 8),style='white');
fig.suptitle(' Rasters after Shift Model (CORRECT  non Pitch Shift  lick releases  07/06/2021 Zola) ', fontsize=10, color='0', y='1')
#plt.title('Rasters after Shift Model (18/03/2021 Zola) ')
plt.show()

fig, axes= rasters(linear_aligned_data, sorted_array, subplots=(5, 8),style='white');
fig.suptitle(' Rasters after Linear Model (CORRECT non Pitch Shift lick releases  07/06/2021 Zola) ', fontsize=10, color='0', y='1')
#make_space_above(axes, topmargin=10)
#plt.title('Rasters after Linear Model (18/03/2021 Zola)')
# fig.tight_layout()
# fig.subplots_adjust(top=10)
plt.show();



fig, axes= rasters(linear_aligned_dataLR, sorted_array, subplots=(5, 8),style='white');
fig.suptitle(' Rasters after Linear Model (ordered by LR onset week 07/06/2021 Zola) ', fontsize=10, color='0', y='1')

#make_space_above(axes, topmargin=10)

#plt.title('Rasters after Linear Model (18/03/2021 Zola)')
# fig.tight_layout()
# fig.subplots_adjust(top=10)
plt.show();

BASE_PATH='D:/Electrophysiological Data/F1702_Zola_Nellie/dynamictimewarping/comdistnoPitchShift/may102021'
#os.mkdir(BASE_PATH)
file_name='alignedDataBlockweekmay102021ShiftModellickreleaseDist'
np.save(os.path.join(BASE_PATH, file_name), shift_aligned_data["spiketimes"])
np.save(os.path.join(BASE_PATH, 'may10neuronIDsnPS'), shift_aligned_data["neurons"])
np.save(os.path.join(BASE_PATH, 'may10trialIDsnPS'), shift_aligned_data["trials"])

file_name='alignedDataBlockweekmay102021LinearModellickreleaseDist'
np.save(os.path.join(BASE_PATH, file_name), linear_aligned_data["spiketimes"])
np.save(os.path.join(BASE_PATH, 'may10linearModelneuronIDsnPS'), linear_aligned_data["neurons"])
np.save(os.path.join(BASE_PATH, 'may10linearModeltrialIDsnPS'), linear_aligned_data["trials"])

file_name='alignedDataBlockweekmay102021OriginalModellickreleaseDist'
np.save(os.path.join(BASE_PATH, file_name), cropped_data2["spiketimes"])
np.save(os.path.join(BASE_PATH, 'may10OriginalModelneuronIDsnPS'), cropped_data2["neurons"])
np.save(os.path.join(BASE_PATH, 'may10OriginalModeltrialIDsnPS'), cropped_data2["trials"])
