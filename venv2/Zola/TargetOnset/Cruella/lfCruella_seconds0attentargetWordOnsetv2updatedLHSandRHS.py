import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping
import os
import h5py
import numpy as np
import scipy.io as rd
import mat73
import seaborn as sns
from func_Cruella_seconds0attentargetWordOnsetv2updatedLHSandRHS import *

#user_input = input('What is the name of your directory')
f={}
blockData={}
#blocksOfInterest=[118, 119,123,126,127,128,129, 135,136, 137,139,140,141,142,143]
blocksOfInterest=[1,2,8,9,10, 11,12,13,14, 15]
blocksOfInterest=[50,51,52,53,54,55,56,57,58,59,60, 61, 62, 63, 64, 65, 676, 67, 68, 69, 70]
left_hand_or_right=['BB2BB3']


pitch_shift_or_not=['nopitchshift', 'pitchshift']
#            highpassfilterParentName=['D:\Electrophysiological Data\F1815_Cruella\HP_' num2str(currBlockName) '\targetword\\orderingbyLRtime\correctresp\' left_hand_or_right{k0}]; %bb4bb5\TARGETonset\rhstim

pitch_shift_or_not=['correctresp']
for k00 in pitch_shift_or_not:
    blocksOfInterest = [ 61, 62, 63, 64, 65, 676, 67, 68, 69, 70]

    blocksOfInterest2 = []
    f = {}
    blockData = {}
    for k0 in left_hand_or_right:
        for i in blocksOfInterest:
            user_input = 'D:/Electrophysiological_Data/F1815_Cruella/LFP_BlockNellie-' + str(
                i) + '/targetword//orderingbyLRtime/' + k00 + '/' + k0 + '/'
            # directory = os.listdir(user_input)

            searchstring = 'Arrays'  # input('What word are you trying to find?')
            if os.path.isdir(user_input) is False:
                print('does not exist')
                blocksOfInterest.remove(i)

            if os.path.isdir(user_input) is True:
                directory = os.listdir(user_input)
                for fname in directory:
                    if searchstring in fname:
                        # Full path
                        f[i] = mat73.loadmat(user_input + os.sep + fname)
                        items = f[i].items()
                        blocksOfInterest2.append(i)

                        arrays = {}
                        for k3, v3 in f[i].items():
                            #newarray3 = (np.array(v3))
                            newarray3=np.asarray(v3).tolist()
                            #newarrayremove3 = newarray3[0, :]
                            arrays[k3] = newarray3
                        blockData[i] = arrays

                        #f[i].close()
        total_lfp=[]
        for key in blockData.keys():
            selected_block=blockData[key]
            selected_lfp=selected_block['lfp_avg_mat_by_trial_cell']
            total_lfp.extend(selected_lfp)
        blocksOfInterest2 = set(blocksOfInterest2)
        blocksOfInterest2 = list(blocksOfInterest2)

        searchstring = 'Arrays'  # input('What word are you trying to find?')
        # for fname in directory:
        #     if searchstring in fname:
        #         # Full path
        #         f[i] = h5py.File(user_input + os.sep + fname)
        #         items = f[i].items()
        #         arrays = {}
        #         for k3, v3 in f[i].items():
        #             newarray3 = np.array(v3)
        #             newarrayremove3 = newarray3[0, :]
        #             arrays[k3] = newarrayremove3
        #         blockData[i] = arrays
        #
        #         f[i].close()

        TMIN = 0 * 1000  # s
        # TMAX = 0.8*1000 # s
        # BINSIZE = 0.01*1000  # 10 ms
        # NBINS = int((TMAX - TMIN) / BINSIZE)

        TMIN2 = 0
        TMAX2 = 1.2;  # I made the maximum trial length 1.2 seconds
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
        # S = dict(np.load("umi_spike_data.npz"))
        # data = SpikeData(
        #     trials=S["trials"],
        #     spiketimes=S["spiketimes"],
        #     neurons=S["unit_ids"],
        #     tmin=TMIN,
        #     tmax=TMAX,
        # )
        # result = arrays["oneDtrialIDarray"];
        # result = x[0, :, 0]
        # adjustedTrial = {}
        # for i2 in range(len(blocksOfInterest2) - 1):
        #     if i2 == 0:
        #         adjustedTrial[i2] = blockData[blocksOfInterest2[i2 + 1]]["oneDtrialIDarray"] + max(
        #             blockData[blocksOfInterest2[i2]]["oneDtrialIDarray"])
        #     else:
        #         adjustedTrial[i2] = blockData[blocksOfInterest2[i2 + 1]]["oneDtrialIDarray"] + max(
        #             adjustedTrial[i2 - 1])
        #
        # if bool(adjustedTrial):
        #     combinedTrialsAdjusted = np.concatenate([v for k, v in sorted(adjustedTrial.items())], 0)
        #     firsttrialarray = blockData[blocksOfInterest2[0]]["oneDtrialIDarray"]
        #     combinedTrials = np.append(firsttrialarray, combinedTrialsAdjusted)
        # else:
        #     combinedTrialsAdjusted = blockData[blocksOfInterest2[0]]["oneDtrialIDarray"]
        #     # firsttrialarray = blockData[blocksOfInterest2[0]]["oneDtrialIDarray"]
        #     combinedTrials = combinedTrialsAdjusted
        #
        # for i in range(len(combinedTrials)):
        #     combinedTrials[i] -= 1
        #
        # combinedSpikeTimes=np.array([]); #declare empty numpy array
        # combinedNeuron=np.array([])
        # combinedLickReleaseTimes=np.array([])
        #
        # for i3 in range(len(blockData)):
        #     selectedSpikeTimes = blockData[blocksOfInterest2[i3]]["oneDspiketimearray"]
        #     selectedNeuronIDs = blockData[blocksOfInterest2[i3]]["oneDspikeIDarray"]
        #     selectedLickReleaseIDs = blockData[blocksOfInterest2[i3]]["oneDlickReleaseArray"]
        #     combinedSpikeTimes = np.append(combinedSpikeTimes, selectedSpikeTimes)
        #     combinedNeuron = np.append(combinedNeuron, selectedNeuronIDs)
        #     combinedLickReleaseTimes = np.append(combinedLickReleaseTimes, selectedLickReleaseIDs)

        #combinedSpikeTimes=np.concatenate([v for k,v in sorted(blockData.items())], key='oneDspiketimearray',  axis=0)
        TMAX =0.8*1000#max(combinedLickReleaseTimes) # s
        BINSIZE = 0.01*1000  # 10 ms
        NBINS = int((TMAX - TMIN) / BINSIZE)
        TMINz=0.2*1000;
        TMAXz =0.29*1000#max(combinedLickReleaseTimes) # s
        BINSIZEz = 0.001*1000 #0.001*1000  # 10 ms
        NBINSz = int((TMAXz - TMINz) / BINSIZEz)

        TMINzb=0.0*1000;
        TMAXzb =0.09*1000#max(combinedLickReleaseTimes) # s
        BINSIZEzb = 0.001*1000 #0.001*1000  # 10 ms
        NBINSzb = int((TMAXzb - TMINzb) / BINSIZEzb)
        #adjustedTrial=arrays2["oneDtrialIDarray"]+max(arrays["oneDtrialIDarray"])
        #adjustedTrial2=arrays3["oneDtrialIDarray"]+max(adjustedTrial)
        # combinedTrials=np.concatenate((arrays["oneDtrialIDarray"], adjustedTrial, adjustedTrial2), axis=0)
        # combinedSpikeTimes=np.concatenate((arrays["oneDspiketimearray"], arrays2["oneDspiketimearray"], arrays3["oneDspiketimearray"]),axis=0)
        # combinedNeuron=np.concatenate((arrays["oneDspikeIDarray"], arrays2["oneDspikeIDarray"], arrays3["oneDspikeIDarray"]),axis=0)

        # data2=SpikeData(
        #     trials=combinedTrials, #arrays["oneDtrialIDarray"],
        #     spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
        #     neurons=combinedNeuron, #["oneDspikeIDarray"],
        #     tmin=TMIN,
        #     tmax=TMAX,
        # )
        #
        #
        # data2.n_neurons=data2.n_neurons.astype(np.int64)
        # data2.n_trials=data2.n_trials.astype(np.int64)
        #
        # data2z=SpikeData(
        #     trials=combinedTrials, #arrays["oneDtrialIDarray"],
        #     spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
        #     neurons=combinedNeuron, #["oneDspikeIDarray"],
        #     tmin=TMINz,
        #     tmax=TMAXz,
        # )
        #
        #
        # data2z.n_neurons=data2z.n_neurons.astype(np.int64)
        # data2z.n_trials=data2z.n_trials.astype(np.int64)
        #
        # data2zb=SpikeData(
        #     trials=combinedTrials, #arrays["oneDtrialIDarray"],
        #     spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
        #     neurons=combinedNeuron, #["oneDspikeIDarray"],
        #     tmin=TMINzb,
        #     tmax=TMAXzb,
        # )
        #
        #
        # data2zb.n_neurons=data2zb.n_neurons.astype(np.int64)
        # data2zb.n_trials=data2zb.n_trials.astype(np.int64)
        # # Bin and normalize (soft z-score) spike times.
        # # Bin and normalize (soft z-score) spike times.
        # binned = data2.bin_spikes(NBINS)
        # binned = binned - binned.mean(axis=(0, 1), keepdims=True)
        # binned = binned / (1e-2 + binned.std(axis=(0, 1), keepdims=True))
        #
        # trialrows= np.array([])
        # maxcombinedTrials=int(max(combinedTrials))
        # for i in range(maxcombinedTrials+1):
        #     #trialrows.append((i)+1)
        #     trialrows=np.append(trialrows,float(i))
        #
        # trialalignment=np.concatenate((combinedLickReleaseTimes.reshape(-1,1),trialrows.reshape(-1,1)),axis=1)
        # #indTrial=np.argsort(trialalignment[:,0])
        # sorted_array = trialalignment[np.argsort(trialalignment[:, 0])]
        # sorted_array_trial=sorted_array[:,1]
        # sorted_array_trial=(sorted_array_trial).astype(np.int)
        # #t3 = np.concatenate((t1.reshape(-1,1),t2.reshape(-1,1),axis=1)
        #
        # #data3=data2.select_trials([1,2,3,4,5])
        # #data4=data3.reorder_trials([0,1,3,2,4])
        #
        # data22=data2.reorder_trials(sorted_array_trial)
        # # Bin and normalize (soft z-score) spike times.
        # binnedLR = data2.bin_spikes(NBINS)
        # binnedLR = binnedLR - binnedLR.mean(axis=(0, 1), keepdims=True)
        # binnedLR = binnedLR / (1e-2 + binnedLR.std(axis=(0, 1), keepdims=True))
        # #binnedLRStdDev=binnedLR.std(axis=(0, 1), keepdims=True)
        #
        # binnedLRz = data2z.bin_spikes(NBINSz)
        # binnedmeans=binnedLRz.mean(axis=(0, 1), keepdims=True)
        #
        # binnedLRz = binnedLRz- binnedLRz.mean(axis=(0, 1), keepdims=True)
        # binnedLRz = binnedLRz / (binnedLRz.std(axis=(0, 1), keepdims=True))
        # binnedLRStdDev=binnedLRz.std(axis=(0, 1), keepdims=True)
        #
        # binnedLRzb = data2zb.bin_spikes(NBINSzb)
        # binnedmeansb=binnedLRzb.mean(axis=(0, 1), keepdims=True)
        #
        # binnedLRzb = binnedLRzb- binnedLRzb.mean(axis=(0, 1), keepdims=True)
        # binnedLRzb = binnedLRzb / (binnedLRzb.std(axis=(0, 1), keepdims=True))
        # binnedLRStdDevb=binnedLRzb.std(axis=(0, 1), keepdims=True)
        #
        # neuronselect=[]
        # neuronselectb=[]
        # meanneuronselect=[]
        # neuronselectmat= np.array([])
        # neuronselectmatb= np.array([])
        # selectedchantoadd=np.array([])
        # neuronsbychan={}
        # neuronsbychanb={}
        #
        # for i in range(len(binnedLRz)):
        #     print(i)
        #     neuronselect=binnedLRz[i]
        #     neuronselectb=binnedLRzb[i]
        #     for i2 in range(1,33):
        #         selectedchantoadd=neuronselect[:,i2]
        #         selectedchantoaddb=neuronselectb[:,i2];
        #         #neuronselect2=neuronselectmat[i2]
        #         neuronselectmat=np.append(neuronselectmat,selectedchantoadd, axis=0)
        #         neuronselectmatb=np.append(neuronselectmatb, selectedchantoaddb, axis=0)
        #         neuronsbychan[i2]=np.mean(neuronselectmat)
        #         neuronsbychanb[i2]=np.mean(neuronselectmatb)
        #         #meanneuronselect[i]=mean(neuronselect[i])
        #
        # goodChanlist=np.array([])
        # binnedmeans=binnedmeans[0]
        # binnedmeans=binnedmeans[0]
        #
        # binnedmeansb=binnedmeansb[0]
        # binnedmeansb=binnedmeansb[0]
        #
        #
        #
        #
        #
        #
        #
        # cropped_data2 = data22.crop_spiketimes(TMIN, TMAX)
        # # Crop spike times when visualizing rasters.
        # cropped_data = data2.crop_spiketimes(TMIN, TMAX)
        #
        # # Load LFP traces (n_trials x n_timebins). Crop traces to [TMIN, TMAX).
        L = dict(np.load("umi_lfp_data.npz"))
        #
        # # Define bandpass filtering function for LFP
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
        # L = dict(np.load("umi_lfp_data.npz"))
        #
        # # Apply bandpass filter.
        # lfp = bandpass(L["lfp"], LOW_CUTOFF, HIGH_CUTOFF, L["sample_rate"])
        #
        # # Crop LFP time base to match spike times.
        # tidx = (L["lfp_time"] >= TMIN) & (L["lfp_time"] < TMAX)
        # lfp = lfp[:, tidx]
        # lfp_time = L["lfp_time"][tidx]
        #
        # # Z-score LFP.
        # lfp -= lfp.mean(axis=1, keepdims=True)
        # lfp /= lfp.std(axis=1, keepdims=True)


        # Specify model.
        # shift_model = ShiftWarping(
        #     smoothness_reg_scale=SHIFT_SMOOTHNESS_REG,
        #     warp_reg_scale=SHIFT_WARP_REG,
        #     maxlag=MAXLAG,
        # )

        # Fit to binned spike times.
        [shift_model, lin_model]=disgustingly_long_func(pitch_shift_or_not, left_hand_or_right, [ 61, 62, 63, 64, 65, 676, 67, 68, 69, 70])
        total_lfp_np=(np.array(total_lfp))
        total_lfp_np=np.mean(total_lfp_np, axis=2)
        start = -0.2
        stoptime = 1.8
        lfp_time = np.linspace(start, stoptime, num=int(24414 * (stoptime - start)))

        tidx = (lfp_time>= TMIN) & (lfp_time < TMAX)
        total_lfp_np = total_lfp_np[:, tidx]
        lfp_time_crop = lfp_time[tidx]

        total_lfp_np /= total_lfp_np.std(axis=1, keepdims=True)
        shift_model_lfp=shift_model.transform(total_lfp_np)[:, :, 0]
        lin_model_lfp=lin_model.transform(total_lfp_np)[:, :, 0]


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

        imkw = dict(clim=(-2, 2), cmap='bwr', interpolation="none", aspect="auto")

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 3.5))

        axes[0].imshow(total_lfp_np, **imkw)
        axes[1].imshow(shift_model_lfp, **imkw)
        axes[2].imshow(lin_model_lfp, **imkw)

        axes[0].set_title("raw lfp (bandpass-filtered)")
        axes[1].set_title("shift aligned")
        axes[2].set_title("linear aligned")

        axes[0].set_ylabel("trials")

        for ax in axes:
            i = np.linspace(0, lfp_time_crop.size - 1, 3).astype(int)
            ax.set_xticks(i)
            ax.set_xticklabels(lfp_time_crop[i])
            ax.set_xlabel("time (ms)")

        fig.tight_layout()
        plt.show()
        import seaborn as sns
        ##adding yticks with the actual lick release time in ms relative to the start trial lick
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 3.5))

        sns.lineplot(axes=axes[0],data=total_lfp_np)
        sns.lineplot(axes=axes[1],data=shift_model_lfp)
        sns.lineplot(axes=axes[2],data=lin_model_lfp)



        axes[0].set_title("raw lfp (bandpass-filtered)")
        axes[1].set_title("shift aligned")
        axes[2].set_title("linear aligned")

        axes[0].set_ylabel("trials")
        plt.show()


        # np.save(os.path.join(BASE_PATH2, file_name), shift_aligned_data["spiketimes"])
        # np.save(os.path.join(BASE_PATH2, 'april042022neuronIDsPS'), shift_aligned_data["neurons"])
        # np.save(os.path.join(BASE_PATH2, 'april042022trialIDsPS'), shift_aligned_data["trials"])
        # file_name='alignedDataBlockweekapril042022LinearModellickrelease'
        #
        # file_name = 'alignedDataBlockweekapril042022LinearModellickrelease'
        # np.save(os.path.join(BASE_PATH2, file_name), linear_aligned_data["spiketimes"])
        # np.save(os.path.join(BASE_PATH2, 'april042022linearModelneuronIDsPS'), linear_aligned_data["neurons"])
        # np.save(os.path.join(BASE_PATH2, 'april042022linearModeltrialIDsPS'), linear_aligned_data["trials"])
        # file_name='alignedDataBlockweekapril042022OriginalModellickrelease'
        #
        # file_name = 'alignedDataBlockweekapril042022OriginalModellickrelease'
        # np.save(os.path.join(BASE_PATH2, file_name), cropped_data2["spiketimes"])
        # np.save(os.path.join(BASE_PATH2, 'april042022OriginalModelneuronIDsPS'), cropped_data2["neurons"])
        # np.save(os.path.join(BASE_PATH2, 'april042022OriginalModeltrialIDsPS'), cropped_data2["trials"])

