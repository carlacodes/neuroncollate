import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping
import os
import h5py
import numpy as np
import scipy


def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


#user_input = input('What is the name of your directory')
f={}
blockData={}
blocksOfInterest=[ 283, 285]
for i in blocksOfInterest:
    user_input = 'D:/Electrophysiological Data/F1704_FloJo/HP_BlockNellie-'+str(i)+'/targetword/nopitchshiftTarget/orderingbyLRtime/nomisses2s'
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

fLFP={}
blockDataLFP={}
LFPBlocktrial={}
LFPBlocktime={}
objMatBlock={}
objTimeBlock={}

#objMat=[]
for i in blocksOfInterest:
    user_input = 'D:/Electrophysiological Data/F1704_FloJo/LFP_BlockNellieL22/weekjuly192021//LFP_Block'+str(i)
    directory = os.listdir(user_input)

    searchstring = 'LFP'+str(i)
                                          #input('What word are you trying to find?')

    for fname in directory:
        if searchstring in fname:
            with h5py.File(user_input + os.sep +fname , 'r') as f:
                for dset in traverse_datasets(f):
                    print('Path:', dset)
                    print('Shape:', f[dset].shape)
                    print('Data type:', f[dset].dtype)

            # Full path
            fLFP[i] = h5py.File(user_input + os.sep + fname)
            test=fLFP[i]['siteTrialMat']
            testTime=fLFP[i]['siteTimeMat']

            range2=test.value.shape
            range2[1]
            objMat = [[0] * (range2[1])] * (32)
            objMatTime=[[0] * (range2[1])] * (32)
            for i0 in range(32):
                for i2 in range((range2[1])):
                    st=test[i0][i2]
                    obj = fLFP[i][st]
                    ob2 = obj[:];
                    stTime=testTime[i0][i2]
                    objtime=fLFP[i][stTime]
                    ob2time=objtime[:]
                    #a.append([0] * m)

                   # objMat.append([0]*i2)
                    objMat[i0][i2]=ob2
                    objMatTime[i0][i2]=ob2time
            objMatBlock[i]=objMat
            objTimeBlock[i]=objMatTime


            # st = test[0][0]
            # obj = fLFP[i][st]
            # ob2=obj[:];
            itemsLFP = fLFP[i].items()
            LFPBlocktrial[i]= fLFP[i]['siteTrialMat']
            LFPBlocktime[i] = fLFP[i]['siteTimeMat']

            # objMat[i]=ob2;

            arraysLFP = {}
            for k3, v3 in fLFP[i].items():
                newarray3LFP = np.array(v3)
                #newarrayremove3LFP = newarray3LFP[0, :]
                arraysLFP[k3] = newarray3LFP
            blockDataLFP[i]=arraysLFP

            fLFP[i].close()

# for i1 in blocksOfInterest:
#     for i2 in range(len(LFPBlocktime)):
#         test1=LFPBlocktrial[i1]
#         objNew=test1[0][0]
#         objNew=objNew[:]
#         objMat[i1][i2]=objNew;





TMIN = 0.2*1000  # s
#TMAX = 0.8*1000 # s
# BINSIZE = 0.01*1000  # 10 ms
# NBINS = int((TMAX - TMIN) / BINSIZE)

TMIN2=0
TMAX2=1.2; #I made the maximum trial length 1.2 seconds
# LFP parameters.
LOW_CUTOFF = 10  # Hz for lfp
HIGH_CUTOFF = 30  # Hz for lfp

# Hyperparameters for shift-only warping model.
SHIFT_SMOOTHNESS_REG = 0.5
SHIFT_WARP_REG = 1e-2
MAXLAG = 0.15

# Hyperparameters for linear warping model.
LINEAR_SMOOTHNESS_REG = 1.0
LINEAR_WARP_REG = 0.065

from affinewarp import SpikeData

# Spike times.
#S = dict(np.load("umi_spike_data.npz"))
# data = SpikeData(
#     trials=S["trials"],
#     spiketimes=S["spiketimes"],
#     neurons=S["unit_ids"],
#     tmin=TMIN,
#     tmax=TMAX,
# )
# result = arrays["oneDtrialIDarray"];
# result = x[0, :, 0]
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
    #data = hf.get('dataset_name').value
    selectedSpikeTimes=blockData[blocksOfInterest[i3]]["oneDspiketimearray"]
    selectedNeuronIDs=blockData[blocksOfInterest[i3]]["oneDspikeIDarray"]
    selectedLickReleaseIDs=blockData[blocksOfInterest[i3]]["oneDlickReleaseArray"]
    combinedSpikeTimes=np.append(combinedSpikeTimes,selectedSpikeTimes)
    combinedNeuron=np.append(combinedNeuron, selectedNeuronIDs)
    combinedLickReleaseTimes=np.append(combinedLickReleaseTimes,selectedLickReleaseIDs)


combinedLFP={}; #declare empty numpy array

#a = np.array([])
# for x in y:
#     a = np.append(a, x)
for i3 in range(len(blockDataLFP)-1):
    selectedLFPsite=blockDataLFP[blocksOfInterest[i3]]["siteTrialMat"][1,:]
    selectedLFPMat=objMatBlock[blocksOfInterest[i3]]
    selectedLFPMat=np.array(selectedLFPMat)
    if i3==0:
        selectedLFP=blockDataLFP[blocksOfInterest[i3]]["toCheck"]
        selectedLFP2=blockDataLFP[blocksOfInterest[i3+1]]["toCheck"]
        selectedLFP3=np.concatenate((selectedLFP,selectedLFP2), axis=1)

        combinedLFPtrials1=selectedLFPMat
        combinedLFPtrials2=np.array(objMatBlock[blocksOfInterest[i3+1]])
        combinedLFPtrials3=np.concatenate((combinedLFPtrials1, combinedLFPtrials2), axis=1)

        selectedtrial=blockDataLFP[blocksOfInterest[i3]]["LFPtimearray"]
        selectedtrial2=blockDataLFP[blocksOfInterest[i3+1]]["LFPtimearray"]
        selectedtrial3=np.concatenate((selectedtrial,selectedtrial2), axis=1)

        combinedTimetrials1 = np.array(objTimeBlock[blocksOfInterest[i3]])
        combinedTimetrials2 =  np.array(objTimeBlock[blocksOfInterest[i3+1]])
        combinedTimetrials3 = np.concatenate((combinedTimetrials1, combinedTimetrials2), axis=1)
    else:
        selectedLFP3=np.concatenate((selectedLFP3, blockDataLFP[blocksOfInterest[i3+1]]["toCheck"]), axis=1)
        selectrial3=np.concatenate((selectedtrial3, blockDataLFP[blocksOfInterest[i3+1]]["LFPtimearray"]), axis=1)
        combinedLFPtrials3=np.concatenate((combinedLFPtrials3,np.array(objMatBlock[blocksOfInterest[i3+1]])), axis=1 )
        combinedTimetrials3 =np.concatenate((combinedTimetrials3,np.array(objTimeBlock[blocksOfInterest[i3+1]])), axis=1)

    selectedLFPsitearray=[];
    # for i4 in range(32):
    #     for i5 in range(len(selectedLFPsite)):
    #         selectedLFPsitearray=selectedLFPsite[i4][1]
    # for i2 in range(len(selectedLFP)):
    #     #combinedLFP = np.append[combinedLFP, i2]
    #     if i3==0:
    #         combinedLFP[i2] = (selectedLFP[i2])
    #
    #     else:
    #
    #         combinedLFP[i2]=np.append(combinedLFP[i2], selectedLFP[i2])

# dataLFP = list(combinedLFP.items())
# an_array = np. array(dataLFP)
# for i22 in range(len(an_array)):
#     an_array[i22]=an_array[i22][1:]
#     an_array[i22]=an_array[i22][0][0]
    # combinedLFPdata = list(combinedLFP.items())
# combinedLFPdata = combinedLFPdata[:][1:]
#
# res = np.array([list(item.values()) for item in combinedLFP.values()])


#combinedSpikeTimes=np.concatenate([v for k,v in sorted(blockData.items())], key='oneDspiketimearray',  axis=0)
TMAX =1*1000#max(combinedLickReleaseTimes) # s
BINSIZE = 0.01*1000  # 10 ms
NBINS = int((TMAX - TMIN) / BINSIZE)
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
#L = dict(np.load("umi_lfp_data.npz"))

# Define bandpass filtering function for LFP
from scipy.signal import butter, filtfilt, freqz

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
#
L = dict(np.load("umi_lfp_data.npz"))



##need to fix this doesn't have a time element
# Apply bandpass filter.

selectedsite=combinedLFPtrials3[6][:]
selectedsite = selectedsite[:, 0, :]
lfp = bandpass(selectedsite, LOW_CUTOFF, HIGH_CUTOFF,1000)

selectedtime=combinedTimetrials3[6][:]
selectedtime=selectedtime[0,0,:]
tidx = (selectedtime >= TMIN) & (selectedtime < TMAX)
lfp = lfp[:,tidx]
lfp_time = selectedtime[tidx]
lfp -= lfp.mean(axis=1, keepdims=True)
lfp /= lfp.std(axis=1, keepdims=True)

fSLFP=1000
#tvec =  np.linspace(0, len(lfp[0]), len(lfp[0]))/ fSLFP

imkw = dict(clim=(-2, 2), cmap='bwr', interpolation="none", aspect="auto")

fig, axes = plt.subplots(1, 1, sharey=True, figsize=(10, 3.5))

axes.imshow(lfp, **imkw)
plt.show()



pop_meanLFP =lfp.mean(axis=0)
tx = np.linspace(TMIN, TMAX, 800)
plt.plot(tx, pop_meanLFP, "-k")
plt.show()

# Crop LFP time base to match spike times.
# tidx = (L["lfp_time"] >= TMIN) & (L["lfp_time"] < TMAX)
# lfp = lfp[:, tidx]
# lfp_time = L["lfp_time"][tidx]



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

# plt.plot(shift_model.loss_hist, label="shift")
# plt.plot(lin_model.loss_hist, label="linear")
# plt.xlabel("Iteration")
# plt.ylabel("Normalized Loss")
# plt.legend()
# plt.tight_layout()
# plt.show()

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
fig.suptitle('Original Data (all lick releases 05/07/2021 FloJo) ', fontsize=10, color='0', y='1')

plt.show() #original data

fig, axes=rasters(cropped_data2,sorted_array, subplots=(5, 8), style='white');
fig.suptitle('Original Data Reorganised (CORRECT releases 05/07/2021 FloJo) ', fontsize=10, color='0', y='1')

plt.show() #original data

fig, axes=rasters(shift_aligned_data, sorted_array, subplots=(5, 8),style='white');
fig.suptitle(' Rasters after Shift Model (CORRECT releases  05/07/2021 FloJo) ', fontsize=10, color='0', y='1')
#plt.title('Rasters after Shift Model (18/03/2021 FloJo) ')
plt.show()

fig, axes= rasters(linear_aligned_data, sorted_array, subplots=(5, 8),style='white');
fig.suptitle(' Rasters after Linear Model (CORRECT releases  05/07/2021 FloJo) ', fontsize=10, color='0', y='1')
#make_space_above(axes, topmargin=10)
#plt.title('Rasters after Linear Model (18/03/2021 FloJo)')
# fig.tight_layout()
# fig.subplots_adjust(top=10)
plt.show();



fig, axes= rasters(linear_aligned_dataLR, sorted_array, subplots=(5, 8),style='white');
fig.suptitle(' Rasters after Linear Model (ordered by LR onset 05/07/2021 FloJo) ', fontsize=10, color='0', y='1')

#make_space_above(axes, topmargin=10)

#plt.title('Rasters after Linear Model (18/03/2021 FloJo)')
# fig.tight_layout()
# fig.subplots_adjust(top=10)
plt.show();
pop_mean = cropped_data2.bin_spikes(NBINS).mean(axis=2) / (BINSIZE * 1e-3)
tx = np.linspace(TMIN, TMAX, NBINS)


pop_mean2 = cropped_data2.bin_spikes(800).mean(axis=2) / (BINSIZE * 1e-3)
tx2 = np.linspace(TMIN, TMAX, 800)

# Show 20 example trials.
fig, axes = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(12, 7))

for k, ax in enumerate(axes.ravel()):
    ax.plot(tx, pop_mean[5+k], "-k")
    ax.set_ylim([-10, 100])
    ax.set_xticks([0 , 200, 400 ,600, 800])
    ax.set_title(str(k))
#make_space_above(axes, topmargin=10)
plt.show()
testChan=pop_mean[5+25]
testChan -= testChan.mean(axis=0, keepdims=True)
testChan /= testChan.std(axis=0, keepdims=True)

testChan2=pop_mean2[5+25]
testChan2 -= testChan2.mean(axis=0, keepdims=True)
testChan2 /= testChan2.std(axis=0, keepdims=True)

plt.plot(tx, testChan, "-b")
tx = np.linspace(TMIN, TMAX, 800)
plt.plot(tx, pop_meanLFP, "-k")
#plt.ylim([-10, 100])
plt.xticks([200, 400, 600, 800, 1000], [0, 200, 400, 600, 800])

plt.title(['TDT', str(26), ' (WARP 24) and LFP of TDT 7 (WARP 3)'])
plt.legend(['Mean spike count', 'Mean LFP'])

plt.xlabel('Time relative to target word onset (ms)')
plt.ylabel('z-score (unitless)')
plt.show()

lagBtSignals=scipy.signal.signaltools.correlate( testChan2, pop_meanLFP)
dt = np.arange(1-800, 800)
recovered_time_shift = dt[lagBtSignals.argmax()]
plt.plot(lagBtSignals)
plt.title(['Correlation between TDT', str(26), ' (WARP 24) and LFP of TDT 7 (WARP 3)'])
plt.xlabel('full discrete cross-correlated samples')
plt.ylabel('correlation score (unitless)')

plt.show()
BASE_PATH='D:/Electrophysiological Data/F1704_FloJo/dynamictimewarping/targetword/july192021'
file_name='alignedDataBlockweekjuly192021ShiftModellickrelease'
np.save(os.path.join(BASE_PATH, file_name), shift_aligned_data["spiketimes"])
np.save(os.path.join(BASE_PATH, 'neuronIDsnPS'), shift_aligned_data["neurons"])
np.save(os.path.join(BASE_PATH, 'trialIDsnPS'), shift_aligned_data["trials"])

file_name='alignedDataBlockweekjuly192021LinearModellickrelease'
np.save(os.path.join(BASE_PATH, file_name), linear_aligned_data["spiketimes"])
np.save(os.path.join(BASE_PATH, 'linearModelneuronIDsnPS'), linear_aligned_data["neurons"])
np.save(os.path.join(BASE_PATH, 'linearModeltrialIDsnPS'), linear_aligned_data["trials"])


file_name='alignedDataBlockweekjuly192021OriginalModellickrelease'
np.save(os.path.join(BASE_PATH, file_name), cropped_data2["spiketimes"])
np.save(os.path.join(BASE_PATH, 'july19OriginalModelneuronIDsnPS'), cropped_data2["neurons"])
np.save(os.path.join(BASE_PATH, 'july19OriginalModeltrialIDsnPS'), cropped_data2["trials"])