import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping
import os
import h5py
import numpy as np
import pickle


#user_input = input('What is the name of your directory')
f={}
blockData={}
#blocksOfInterest=[118, 119,123,126,127,128,129, 135,136, 137,139,140,141,142,143]
import matlab
import matlab.engine
import csv
import pandas as pd
import scipy.io
import mat73
import numpy_indexed as npi


eng = matlab.engine.start_matlab()
#
# blocksOfInterest=[178,179,180,181,182,183,184,185]
# left_hand_or_right=['BB2BB3'] ##'BB2BB3'
# for k0 in left_hand_or_right:
#     for i in blocksOfInterest:
#         #        highpassfilterParentName=['D:\Electrophysiological Data\F1702_Zola_Nellie\HP_' num2str(currBlockName) '\\bothstim\orderingbyLRtime\300msepoch']; %BB2BB3\TARGETonset\rhstim
#
#         user_input = 'D:/Electrophysiological Data/F1702_Zola_Nellie/HP_BlockNellie-'+str(i)+'/bothstim/orderingbyLRtime/300msepoch/'
#         directory = os.listdir(user_input)
#
#         searchstring = 'Arrays'#input('What word are you trying to find?')
#
#         for fname in directory:
#             if searchstring in fname:
#                 # Full path
#                 f[i] = h5py.File(user_input + os.sep + fname)
#                 items = f[i].items()
#                 arrays = {}
#                 for k3, v3 in f[i].items():
#                     newarray3 = np.array(v3)
#                     newarrayremove3 = newarray3[0, :]
#                     arrays[k3] = newarrayremove3
#                 blockData[i]=arrays
#
#
#
#                 f[i].close()
#
#
#     TMIN = 0*1000  # s
#     #TMAX = 0.8*1000 # s
#     # BINSIZE = 0.01*1000  # 10 ms
#     # NBINS = int((TMAX - TMIN) / BINSIZE)
#
#     TMIN2=0
#     TMAX2=1.2; #I made the maximum trial length 1.2 seconds
#     # LFP parameters.
#     LOW_CUTOFF = 10  # Hz
#     HIGH_CUTOFF = 30  # Hz
#
#     # Hyperparameters for shift-only warping model.
#     SHIFT_SMOOTHNESS_REG = 0.5
#     SHIFT_WARP_REG = 1e-2
#     MAXLAG = 0.15
#
#     # Hyperparameters for linear warping model.
#     LINEAR_SMOOTHNESS_REG = 1.0
#     LINEAR_WARP_REG = 0.065
#
#     from affinewarp import SpikeData
#
#     # Spike times.
#     #S = dict(np.load("umi_spike_data.npz"))
#     # data = SpikeData(
#     #     trials=S["trials"],
#     #     spiketimes=S["spiketimes"],
#     #     neurons=S["unit_ids"],
#     #     tmin=TMIN,
#     #     tmax=TMAX,
#     # )
#     # result = arrays["oneDtrialIDarray"];
#     # result = x[0, :, 0]
#     adjustedTrial={}
#     for i2 in range(len(blocksOfInterest)-1):
#         if i2==0:
#             adjustedTrial[i2]=blockData[blocksOfInterest[i2+1]]["oneDtrialIDarray"]+max(blockData[blocksOfInterest[i2]]["oneDtrialIDarray"])
#         else:
#             adjustedTrial[i2]=blockData[blocksOfInterest[i2+1]]["oneDtrialIDarray"]+max(adjustedTrial[i2-1])
#
#     combinedTrialsAdjusted=np.concatenate([v for k,v in sorted(adjustedTrial.items())], 0)
#     firsttrialarray=blockData[blocksOfInterest[0]]["oneDtrialIDarray"]
#     combinedTrials=np.append(firsttrialarray, combinedTrialsAdjusted)
#     for i in range(len(combinedTrials)):
#         combinedTrials[i] -= 1
#
#     combinedSpikeTimes=np.array([]); #declare empty numpy array
#     combinedNeuron=np.array([])
#     combinedLickReleaseTimes=np.array([])
#     combined_stim_times=np.array([])
#     combined_stim_durs=np.array([])
#     combined_stim_types=np.array([])
#
#     for i3 in range(len(blockData)):
#         selectedSpikeTimes=blockData[blocksOfInterest[i3]]["oneDspiketimearray"]
#         selectedNeuronIDs=blockData[blocksOfInterest[i3]]["oneDspikeIDarray"]
#         selectedLickReleaseIDs=blockData[blocksOfInterest[i3]]["oneDlickReleaseArray"]
#         selected_stim_times=blockData[blocksOfInterest[i3]]["oneDstimIDarray"]
#         selected_stim_durs=blockData[blocksOfInterest[i3]]["oneDstimDurArray"]
#         selected_stim_type=blockData[blocksOfInterest[i3]]["oneDstimTypeArray"]
#
#         combinedSpikeTimes=np.append(combinedSpikeTimes,selectedSpikeTimes)
#         combinedNeuron=np.append(combinedNeuron, selectedNeuronIDs)
#         combinedLickReleaseTimes=np.append(combinedLickReleaseTimes,selectedLickReleaseIDs)
#         combined_stim_times=np.append(combined_stim_times, selected_stim_times)
#         combined_stim_durs=np.append(combined_stim_durs, selected_stim_durs)
#         combined_stim_types=np.append(combined_stim_types, selected_stim_type)
#
#     #combinedSpikeTimes=np.concatenate([v for k,v in sorted(blockData.items())], key='oneDspiketimearray',  axis=0)
#     TMAX =6.3*1000#max(combinedLickReleaseTimes) # s
#     BINSIZE = 0.01*1000  # 10 ms
#     NBINS = int((TMAX - TMIN) / BINSIZE)
#     TMINz=0.2*1000;
#     TMAXz =0.29*1000#max(combinedLickReleaseTimes) # s
#     BINSIZEz = 0.001*1000 #0.001*1000  # 10 ms
#     NBINSz = int((TMAXz - TMINz) / BINSIZEz)
#
#     TMINzb=0.0*1000;
#     TMAXzb =0.09*1000#max(combinedLickReleaseTimes) # s
#     BINSIZEzb = 0.001*1000 #0.001*1000  # 10 ms
#     NBINSzb = int((TMAXzb - TMINzb) / BINSIZEzb)
#     #adjustedTrial=arrays2["oneDtrialIDarray"]+max(arrays["oneDtrialIDarray"])
#     #adjustedTrial2=arrays3["oneDtrialIDarray"]+max(adjustedTrial)
#     # combinedTrials=np.concatenate((arrays["oneDtrialIDarray"], adjustedTrial, adjustedTrial2), axis=0)
#     # combinedSpikeTimes=np.concatenate((arrays["oneDspiketimearray"], arrays2["oneDspiketimearray"], arrays3["oneDspiketimearray"]),axis=0)
#     # combinedNeuron=np.concatenate((arrays["oneDspikeIDarray"], arrays2["oneDspikeIDarray"], arrays3["oneDspikeIDarray"]),axis=0)
#
#     data2=SpikeData(
#         trials=combinedTrials, #arrays["oneDtrialIDarray"],
#         spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
#         neurons=combinedNeuron, #["oneDspikeIDarray"],
#         tmin=TMIN,
#         tmax=TMAX,
#     )
#
#
#     data2.n_neurons=data2.n_neurons.astype(np.int64)
#     data2.n_trials=data2.n_trials.astype(np.int64)
#
#     data2z=SpikeData(
#         trials=combinedTrials, #arrays["oneDtrialIDarray"],
#         spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
#         neurons=combinedNeuron, #["oneDspikeIDarray"],
#         tmin=TMINz,
#         tmax=TMAXz,
#     )
#
#
#     data2z.n_neurons=data2z.n_neurons.astype(np.int64)
#     data2z.n_trials=data2z.n_trials.astype(np.int64)
#
#     data2zb=SpikeData(
#         trials=combinedTrials, #arrays["oneDtrialIDarray"],
#         spiketimes=combinedSpikeTimes, #["oneDspiketimearray"],
#         neurons=combinedNeuron, #["oneDspikeIDarray"],
#         tmin=TMINzb,
#         tmax=TMAXzb,
#     )
#
#
#     data2zb.n_neurons=data2zb.n_neurons.astype(np.int64)
#     data2zb.n_trials=data2zb.n_trials.astype(np.int64)
#     # Bin and normalize (soft z-score) spike times.
#     # Bin and normalize (soft z-score) spike times.
#     binned = data2.bin_spikes(NBINS)
#     binned = binned - binned.mean(axis=(0, 1), keepdims=True)
#     binned = binned / (1e-2 + binned.std(axis=(0, 1), keepdims=True))
#
#     trialrows= np.array([])
#     maxcombinedTrials=int(max(combinedTrials))
#     for i in range(maxcombinedTrials+1):
#         #trialrows.append((i)+1)
#         trialrows=np.append(trialrows,float(i))
#
#     trialalignment=np.concatenate((combinedLickReleaseTimes.reshape(-1,1),trialrows.reshape(-1,1)),axis=1)
#     #indTrial=np.argsort(trialalignment[:,0])
#     sorted_array = trialalignment[np.argsort(trialalignment[:, 0])]
#     sorted_array_trial=sorted_array[:,1]
#     sorted_array_trial=(sorted_array_trial).astype(np.int)
#     #t3 = np.concatenate((t1.reshape(-1,1),t2.reshape(-1,1),axis=1)
#
#     #data3=data2.select_trials([1,2,3,4,5])
#     #data4=data3.reorder_trials([0,1,3,2,4])
#
#     data22=data2.reorder_trials(sorted_array_trial)
#     # Bin and normalize (soft z-score) spike times.
#     binnedLR = data2.bin_spikes(NBINS)
#     binnedLR = binnedLR - binnedLR.mean(axis=(0, 1), keepdims=True)
#     binnedLR = binnedLR / (1e-2 + binnedLR.std(axis=(0, 1), keepdims=True))
#     #binnedLRStdDev=binnedLR.std(axis=(0, 1), keepdims=True)
#
#     binnedLRz = data2z.bin_spikes(NBINSz)
#     binnedmeans=binnedLRz.mean(axis=(0, 1), keepdims=True)
#
#     binnedLRz = binnedLRz- binnedLRz.mean(axis=(0, 1), keepdims=True)
#     binnedLRz = binnedLRz / (binnedLRz.std(axis=(0, 1), keepdims=True))
#     binnedLRStdDev=binnedLRz.std(axis=(0, 1), keepdims=True)
#
#     binnedLRzb = data2zb.bin_spikes(NBINSzb)
#     binnedmeansb=binnedLRzb.mean(axis=(0, 1), keepdims=True)
#
#     binnedLRzb = binnedLRzb- binnedLRzb.mean(axis=(0, 1), keepdims=True)
#     binnedLRzb = binnedLRzb / (binnedLRzb.std(axis=(0, 1), keepdims=True))
#     binnedLRStdDevb=binnedLRzb.std(axis=(0, 1), keepdims=True)
#
#     neuronselect=[]
#     neuronselectb=[]
#     meanneuronselect=[]
#     neuronselectmat= np.array([])
#     neuronselectmatb= np.array([])
#     selectedchantoadd=np.array([])
#     neuronsbychan={}
#     neuronsbychanb={}
#
#     for i in range(len(binnedLRz)):
#         print(i)
#         neuronselect=binnedLRz[i]
#         neuronselectb=binnedLRzb[i]
#         for i2 in range(1,33):
#             selectedchantoadd=neuronselect[:,i2]
#             selectedchantoaddb=neuronselectb[:,i2];
#             #neuronselect2=neuronselectmat[i2]
#             neuronselectmat=np.append(neuronselectmat,selectedchantoadd, axis=0)
#             neuronselectmatb=np.append(neuronselectmatb, selectedchantoaddb, axis=0)
#             neuronsbychan[i2]=np.mean(neuronselectmat)
#             neuronsbychanb[i2]=np.mean(neuronselectmatb)
#             #meanneuronselect[i]=mean(neuronselect[i])
#
#     goodChanlist=np.array([])
#     binnedmeans=binnedmeans[0]
#     binnedmeans=binnedmeans[0]
#
#     binnedmeansb=binnedmeansb[0]
#     binnedmeansb=binnedmeansb[0]
#
#     for i in range(0,len(neuronsbychan)):
#         keys_list = list(neuronsbychan)
#         key = keys_list[i]
#         selectedChan=neuronsbychan[key]
#         counter=binnedLRStdDev[:,:,i]
#         countermean=np.mean(binnedLRStdDev[:,:,1:33])
#         selectedmeans=binnedmeans[ int(key)]
#         selectedmeansb=binnedmeansb[int(key)]
#
#         bigMean=np.mean(binnedmeans[1:33])
#         counter=np.squeeze(countermean)[()]
#         # counter=float(counter[1])
#         # print(counter)
#         if selectedmeans<0:
#             if (selectedmeans)<=(bigMean): #selectedmeans-counter: #-(0.1*(counter)):
#                 print('something good')
#                 goodChanlist=np.append(goodChanlist, int(key))
#         else:
#             if (selectedmeans) >= abs(1.5*selectedmeansb):  # selectedmeans-counter: #-(0.1*(counter)):
#                 print('something good')
#                 goodChanlist = np.append(goodChanlist, int(key))
#
#
#
#
#
#     cropped_data2 = data22.crop_spiketimes(TMIN, TMAX)
#     # Crop spike times when visualizing rasters.
#     cropped_data = data2.crop_spiketimes(TMIN, TMAX)
#
#     # Load LFP traces (n_trials x n_timebins). Crop traces to [TMIN, TMAX).
#     #L = dict(np.load("umi_lfp_data.npz"))
#
#     # Define bandpass filtering function for LFP
#     from scipy.signal import butter, filtfilt, freqz
#
#     def bandpass(x, lowcut, highcut, fs, order=5, axis=-1, kind='butter'):
#         """
#         Bandpass filters analog time series.
#
#         Parameters
#         ----------
#         x : ndarray
#             Time series data
#         lowcut : float
#             Defines lower frequency cutoff (e.g. in Hz)
#         highcut : float
#             Defines upper frequency cutoff (e.g. in Hz)
#         fs : float
#             Sampling frequency (e.g. in Hz)
#         order : int
#             Filter order parameter
#         kind : str
#             Specifies the kind of filter
#         axis : int
#             Axis along which to bandpass filter data
#         """
#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq
#         if kind == "butter":
#             b, a = butter(order, [low, high], btype="band")
#         else:
#             raise ValueError("Filter kind not recognized.")
#         return filtfilt(b, a, x, axis=axis)
#
#     # Load LFP.
#     # L = dict(np.load("umi_lfp_data.npz"))
#     #
#     # # Apply bandpass filter.
#     # lfp = bandpass(L["lfp"], LOW_CUTOFF, HIGH_CUTOFF, L["sample_rate"])
#     #
#     # # Crop LFP time base to match spike times.
#     # tidx = (L["lfp_time"] >= TMIN) & (L["lfp_time"] < TMAX)
#     # lfp = lfp[:, tidx]
#     # lfp_time = L["lfp_time"][tidx]
#     #
#     # # Z-score LFP.
#     # lfp -= lfp.mean(axis=1, keepdims=True)
#     # lfp /= lfp.std(axis=1, keepdims=True)
#
#
#     # Specify model.
#     shift_model = ShiftWarping(
#         smoothness_reg_scale=SHIFT_SMOOTHNESS_REG,
#         warp_reg_scale=SHIFT_WARP_REG,
#         maxlag=MAXLAG,
#     )
#
#     # Fit to binned spike times.
#     shift_model.fit(binned, iterations=50)
#
#     # Apply inverse warping functions to data.
#     shift_aligned_data = shift_model.transform(data2).crop_spiketimes(TMIN, TMAX)
#
#     from affinewarp import PiecewiseWarping
#
#     # Specify model.
#     lin_model = PiecewiseWarping(
#         n_knots=0,
#         smoothness_reg_scale=LINEAR_SMOOTHNESS_REG,
#         warp_reg_scale=LINEAR_WARP_REG
#     )
#
#     # Fit to binned spike times.
#     lin_model.fit(binned, iterations=50)
#
#     # Apply inverse warping functions to data.
#     linear_aligned_data = lin_model.transform(data2).crop_spiketimes(TMIN, TMAX)
#     #trialrows=np.array([ i in (max((combinedTrials)))])
#
#     lin_model.fit(binnedLR, iterations=50)
#
#     # Apply inverse warping functions to data.
#     linear_aligned_dataLR = lin_model.transform(data22).crop_spiketimes(TMIN, TMAX)
#
#     plt.plot(shift_model.loss_hist, label="shift")
#     plt.plot(lin_model.loss_hist, label="linear")
#     plt.xlabel("Iteration")
#     plt.ylabel("Normalized Loss")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     def make_space_above(axes, topmargin=1):
#         """ increase figure size to make topmargin (in inches) space for
#             titles, without changing the axes sizes"""
#         fig = axes.flatten()[0].figure
#         s = fig.subplotpars
#         w, h = fig.get_size_inches()
#
#         figh = h - (1-s.top)*h  + topmargin
#         fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
#         fig.set_figheight(figh)
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     ##adding yticks with the actual lick release time in ms relative to the start trial lick
#
#     from visualization03022022 import psth_plots, rasters, rasterSite, psthind
#
#     epoch_offset = 300
#     # fig, axes = rasters(cropped_data, sorted_array, epoch_offset, (5, 8), style='white');
#     # fig.suptitle('Original Data (all lick releases 12/07/2021 Zola) ', fontsize=10, color='0', y='1')
#     #
#     # plt.show()  # original data
#
#     fig, axes = rasters(cropped_data2, sorted_array, epoch_offset, subplots=(5, 8), style='white');
#     fig.suptitle('Original Data Reorganised by Lick Release, Aligned to Sound Onset week 12/07/2021, Zola) ',
#                  fontsize=10, color='0', y='1')
#
#     plt.show()  # original data
#
#     fig, axes = rasters(linear_aligned_dataLR, sorted_array, epoch_offset, subplots=(5, 8), style='white');
#     fig.suptitle(' Rasters after Linear Model (ordered by LR onset 12/07/2021 Zola) ', fontsize=10, color='0', y='1')
#
#     fig2, axes2 = psth_plots(cropped_data2, sorted_array, NBINS, TMIN, TMAX, combinedTrials, 'purple', epoch_offset,
#                              subplots=(5, 8), style='white');
#     fig2.suptitle(' PSTHs (week 12/07/2021 Zola) ', fontsize=10, color='0', y='1')
#     plt.show()
#     # (3,13,14,15, 17, 27)
#     fig= rasterSite(cropped_data2, sorted_array,[3,13,14,15, 17, 27], epoch_offset, style='black');
#
#     fig3=psthind(cropped_data2, sorted_array, NBINS, TMIN, TMAX, combinedTrials, 'purple', epoch_offset,[3,13,14,15, 17, 27])
#     #fig= rasterSite(cropped_data2, sorted_array,[13], epoch_offset, style='black');
#
#     #plt.show()
#
# #def rasterSite(data,sorted_array, siteschosen, fig=None, max_spikes=7000, style='black', **scatter_kw):
#
#
#
#
#     #make_space_above(axes, topmargin=10)
#
#     #plt.title('Rasters after Linear Model (18/03/2021 Zola)')
#     # fig.tight_layout()
#     # fig.subplots_adjust(top=10)
#     plt.show();
#     BASE_PATH2 = 'D:/Electrophysiological Data/F1702_Zola/dynamictimewarping/l27soundonset/'+k0+'/'
#     if os.path.isdir(BASE_PATH2) is False:
#         os.makedirs(BASE_PATH2)
#     # np.save(os.path.join(BASE_PATH2, file_name), shift_aligned_data["spiketimes"])
#     # np.save(os.path.join(BASE_PATH2, 'january3122neuronIDsPS'), shift_aligned_data["neurons"])
#     # np.save(os.path.join(BASE_PATH2, 'january3122trialIDsPS'), shift_aligned_data["trials"])
#     #
#     # file_name = 'alignedDataBlockweekjanuary312022LinearModellickrelease'
#     # np.save(os.path.join(BASE_PATH2, file_name), linear_aligned_data["spiketimes"])
#     # np.save(os.path.join(BASE_PATH2, 'january3122linearModelneuronIDsPS'), linear_aligned_data["neurons"])
#     # np.save(os.path.join(BASE_PATH2, 'january3122linearModeltrialIDsPS'), linear_aligned_data["trials"])
#     #
#     file_name = 'alignedDataBlockweekjuly122021OriginalModellickrelease'
#     np.save(os.path.join(BASE_PATH2, file_name), cropped_data2["spiketimes"])
#     np.save(os.path.join(BASE_PATH2, 'july122021OriginalModelneuronIDsPS'), cropped_data2["neurons"])
#     np.save(os.path.join(BASE_PATH2, 'july122021OriginalModeltrialIDsPS'), cropped_data2["trials"])
#     np.save(os.path.join(BASE_PATH2, 'july122021stim_times'), combined_stim_times)
#     np.save(os.path.join(BASE_PATH2, 'july122021stim_durs'), combined_stim_durs)
#     np.save(os.path.join(BASE_PATH2, 'july122021stim_types'), combined_stim_types)

    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata'
    # np.save(os.path.join(BASE_PATH2, file_name), cropped_data2, allow_pickle=True)
    #
    # with open(os.path.join(BASE_PATH2, file_name), 'wb') as f:
    #     pickle.dump(cropped_data2, f)
    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdataNbins'
    # np.save(os.path.join(BASE_PATH2, file_name), NBINS)
    # #TMIN, TMAX, sorted_array, combinedTrials, epoch_offset
    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata_tmin'
    # np.save(os.path.join(BASE_PATH2, file_name), TMIN)
    #
    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata_tmax'
    # np.save(os.path.join(BASE_PATH2, file_name), TMAX)
    #
    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata_sortedarray'
    # np.save(os.path.join(BASE_PATH2, file_name), sorted_array)
    #
    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata_combinedtrials'
    # np.save(os.path.join(BASE_PATH2, file_name), combinedTrials)
    #
    # file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata_epochoffset'
    # np.save(os.path.join(BASE_PATH2, file_name), epoch_offset)
    #
    #
bin_folder = 'D:/Electrophysiological Data/bindata/F1702_Zola/intratrialroving/correctresponse300ms/'
# eng.addpath(eng.genpath('D:/npy-matlab-master'));

rootZsave = 'D:/Electrophysiological Data/bindata/F1702_Zola/intratrialroving/05042022/minus300mstopositive300ms'
# cd(rootZsave);
# spike_times = eng.double(eng.readNPY('spike_times.npy'));
# spike_cluster_IDs = eng.readNPY('spike_clusters.npy');
# spike_goodness = eng.tdfread('cluster_info.tsv');
# channel_map = eng.readNPY('channel_map.npy');
# templates = eng.readNPY('templates.npy');
# templates_ind = eng.readNPY('templates_ind.npy');

spike_times = np.load(os.path.join(rootZsave, 'spike_times.npy'))
spike_cluster_IDs = np.load(os.path.join(rootZsave, 'spike_clusters.npy'))

# spike_goodness = eng.tdfread('cluster_info.tsv');
tsv_file = open(os.path.join(rootZsave, 'cluster_info.tsv'))
spike_goodness = csv.reader(tsv_file, delimiter="\t")
spike_goodness = pd.read_csv(os.path.join(rootZsave, 'cluster_info.tsv'), sep='\t')

channel_map = np.load(os.path.join(rootZsave, 'channel_map.npy'))

templates = np.load(os.path.join(rootZsave, 'templates.npy'))
templates_ind = np.load(os.path.join(rootZsave, 'templates_ind.npy'))
#spike_templates = double(readNPY(fullfile(rootZsave,'spike_templates.npy')));

spike_templates=np.load(os.path.join(rootZsave, 'spike_templates.npy'))
array_mua = np.array([]);
for i in range(0, len(spike_goodness['cluster_id'])):
    array_mua_test = (spike_goodness['KSLabel'][i])
    if array_mua_test== 'good':

        array_mua_bin = 1;
    else:
        array_mua_bin = 0;

    array_mua = np.append(array_mua, array_mua_bin)

    searchstring = 'trial_map'  # input('What word are you trying to find?')
##         user_input = 'D:/Electrophysiological Data/F1702_Zola_Nellie/HP_BlockNellie-'+str(i)+'/bothstim/orderingbyLRtime/300msepoch/'
    directory = os.listdir(bin_folder)
    for fname in directory:
        if searchstring in fname:
            # Full path
            trial_map = mat73.loadmat(bin_folder + os.sep + fname)
cl_ids=spike_goodness['cluster_id']
nclust = len(cl_ids);

cl_chans = np.array([])
cl_pos = np.array([])
channel_key=pd.Series(spike_goodness['ch'])
array_mua=pd.Series(array_mua)
concatenated_dataframes = pd.concat([channel_key, array_mua], axis=1)

#
# for id in range(0,nclust):
#     uid = cl_ids[id];
#     clidx = find(spike_goodness['id'] == uid);
#     b = index[new_val]
#
#     zchan = spike_goodness['ch'][clidx] + 1;
#     #dpth = clu_info.depth(clidx);
#     cl_chans = np.append(cl_chans, zchan);
#     #cl_pos=np.append(cl_pos, dpth)
trial_map_across_directory=(trial_map['trial_map_across_directory'])
epoch_map_acrossdirectory=trial_map['epoch_map_acrossdirectory']
time_to_targlist_acrossdirec=trial_map['time_to_targlist_acrossdirec']
def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


bin_spks_by_id = {}

for i2 in range(0, len(concatenated_dataframes)):
    bin_spks = {}

    bin_spks_mat = []
    [index_forspiketimes_row] = np.where(spike_cluster_IDs == cl_ids[i2]);
    # index_forspiketimes_row=np.asarray(index_forspiketimes_row).T

    corresponding_spike_times = spike_times[index_forspiketimes_row];
    # [C,ia,ib] = intersect(corresponding_spike_times, epoch_map_acrossdirectory, 'rows');

    common_elements, ar1_i, ar2_i = np.intersect1d(corresponding_spike_times, epoch_map_acrossdirectory,
                                                   return_indices=True)
    # result *= (A[:, 0] == B[:, 0]) * 2 - 1
    # [C, ia, ib] = intersect2D(corresponding_spike_times, epoch_map_acrossdirectory, 'rows');
    correspondingtrials = trial_map_across_directory[ar2_i]
    correspondingtrials[:, 3] = np.round(correspondingtrials[:, 3])

    corresponding_spike_times = spike_times[index_forspiketimes_row];
    # spike_time_start = np.concatenate((correspondingtrials[:, 3], correspondingtrials[:, 4]));
    counter = 0;
    epoch_start = np.unique(correspondingtrials[:, 3])
    epoch_end = np.unique(correspondingtrials[:, 4])
    for i3 in range(0, len(correspondingtrials[:, 3])):
        epoch_1 = (epoch_start[counter]);
        epoch_2 = (epoch_end[counter]);
        if epoch_1 <= (correspondingtrials[i3, 3]) and correspondingtrials[i3, 3] <= epoch_2:
            time_diff = ((correspondingtrials[i3, 3]) - (epoch_1));
            bin_spks_mat = np.append(bin_spks_mat, time_diff)
            bin_spks[counter] = bin_spks_mat
        else:
            bin_spks_mat = [];
            counter = counter + 1;
            time_diff = (correspondingtrials[i3, 3]) - (epoch_1)

            bin_spks_mat = np.append(bin_spks_mat, time_diff)
            bin_spks[counter] = bin_spks_mat

    bin_spks_by_id[i2] = bin_spks
##note that cluster_id values are missing 4 AND 8 as in terms of cluster ids for the Zola -300 ms to +300ms relative to trial
# start and stop data, thus everything is -2 relative to the ID number