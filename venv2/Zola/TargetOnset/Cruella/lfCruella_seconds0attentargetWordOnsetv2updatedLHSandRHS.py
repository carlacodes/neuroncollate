import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping
import os
import h5py
import numpy as np
import scipy.io as rd
import mat73
import seaborn as sns
import scipy
from scipy.signal import butter, lfilter
from func_Cruella_seconds0attentargetWordOnsetv2updatedLHSandRHS import *

#user_input = input('What is the name of your directory')
f={}
blockData={}

left_hand_or_right=['BB2BB3']

pitch_shift_or_not=['correctresp']
for k00 in pitch_shift_or_not:
    blocksOfInterest = [92,93,94,95,96,97,98,99,100,101,102]

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


        # def butter_bandpass(lowcut, highcut, fs, order=5):
        #     return butter(order, [lowcut, highcut], fs=fs, btype='band')
        #
        #
        # def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1):
        #     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        #     y = lfilter(b, a, data, axis=axis)
        #     return y
        #


        # Fit to binned spike times.
        [shift_model, lin_model]=disgustingly_long_func(pitch_shift_or_not, left_hand_or_right, blocksOfInterest2)
        total_lfp_np=(np.array(total_lfp))
        total_lfp_np=np.mean(total_lfp_np, axis=2)

        fs=np.round(24414.0625*(1000/24414))

        ##making copy, z-scoring, then getting model fit
        total_lfp_modelfit=(np.array(total_lfp))
        #repeating S6 word for word, so I need to take the mean across channels and then add an extra dimension to make one "unit" for the spikedata object.


        for k in range(0, 32):
            print(k)
            chosensite=total_lfp_modelfit[:,:, k]
            corresp_bp=bandpass(chosensite,  1,10, fs)
            total_lfp_modelfit[:,:, k]=corresp_bp

        total_lfp_modelfit=np.mean(total_lfp_modelfit, axis=2)

        total_lfp_modelfit /= total_lfp_modelfit.std(axis=(1), keepdims=True)
        plt.plot( total_lfp_modelfit)
        plt.show()
        total_lfp_modelfit=total_lfp_modelfit[:,:, np.newaxis]
        plt.plot(total_lfp_modelfit[:,:, 0])
        plt.show()


        start = 0
        stoptime = 2
        #fs=np.round(24414.0625*(1000/24414))
        lfp_time = np.linspace(start*fs*1000, stoptime*fs*1000, num=int(fs * (stoptime - start)+1))
        start_crop = 0*1000
        stoptime_crop = 0.8*1000

        tidx = (lfp_time>= start_crop*fs) & (lfp_time <= stoptime_crop*fs)
        total_lfp_np = total_lfp_np[:, tidx]
        total_lfp_np=bandpass(total_lfp_np, 5, 9, fs)
        
        
        total_lfp_modelfit = total_lfp_modelfit[:,tidx, :]
        lfp_time_crop = lfp_time[tidx]

        total_lfp_np /= total_lfp_np.std(axis=1, keepdims=True)

        shift_model_lfp=shift_model.transform(total_lfp_np)[:, :, 0]
        lin_model_lfp=lin_model.transform(total_lfp_np)[:, :, 0]


        SHIFT_SMOOTHNESS_REG = 0.5
        SHIFT_WARP_REG = 1e-2
        MAXLAG = 0.15

        LINEAR_SMOOTHNESS_REG = 1.0
        LINEAR_WARP_REG = 0.065


        shift_model_on_lfp = ShiftWarping(
            smoothness_reg_scale=SHIFT_SMOOTHNESS_REG,
            warp_reg_scale=SHIFT_WARP_REG,
            maxlag=MAXLAG,

        )
        from affinewarp import PiecewiseWarping

        lin_model_on_lfp = PiecewiseWarping(
            n_knots=0,
            smoothness_reg_scale=LINEAR_SMOOTHNESS_REG,
            warp_reg_scale=LINEAR_WARP_REG
        )

        # Fit my silly model to the LFP and ideally then compare the spike times to the LFP
        shift_model_on_lfp.fit(total_lfp_modelfit, iterations=50)
        lfp_sm_transf=shift_model_on_lfp.transform(total_lfp_modelfit)[:, :, 0]

        lin_model_on_lfp.fit(total_lfp_modelfit, iterations=50)


        import numpy as np
        import matplotlib.pyplot as plt

        imkw = dict(clim=(-2, 2), cmap='bwr', interpolation="none", aspect="auto")

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 3.5))

        axes[0].imshow(total_lfp_np, **imkw)
        axes[1].imshow(shift_model_lfp, **imkw)
        axes[2].imshow(lin_model_lfp, **imkw)

        axes[0].set_title("raw lfp (bandpass-filtered and then z-scored)")
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

        fig2, axes2 = plt.subplots(1, 3, sharey=True, figsize=(10, 3.5))
        lfp_np_plt=np.mean(total_lfp_np, axis=0)
        shift_model_lfp_plt=np.mean(shift_model_lfp, axis=0)
        lin_model_lfp_plt=np.mean(lin_model_lfp, axis=0)



        axes[0].set_title("raw lfp (bandpass-filtered)")
        axes[1].set_title("shift aligned")
        axes[2].set_title("linear aligned")

        axes[0].set_ylabel("trials")
        plt.show()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle('LFP for f1815 cruella')

        # Bulbasaur
        sns.lineplot(ax=axes[0], x=lfp_time_crop, y=lfp_np_plt)
        sns.lineplot(ax=axes[1], x=lfp_time_crop, y=shift_model_lfp_plt)
        sns.lineplot(ax=axes[2], x=lfp_time_crop, y=lin_model_lfp_plt)
        axes[0].set_title("raw lfp (bandpass-filtered)")
        axes[1].set_title("shift aligned")
        axes[2].set_title("linear aligned")

        axes[0].set_ylabel("a.u.")
        plt.show()
        fractional_shifts_spk_warp=shift_model.fractional_shifts
        fractional_shifts_lfp_warp=shift_model_on_lfp.fractional_shifts
        fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
        corrcoeff=scipy.stats.pearsonr(fractional_shifts_spk_warp, fractional_shifts_lfp_warp)
        sns.scatterplot(x=fractional_shifts_spk_warp, y=fractional_shifts_lfp_warp)
        plt.show()
        #
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
        #
        
        fig2, axes2 = plt.subplots(1, 2, sharey=True, figsize=(10, 3.5))
        axes2[0].imshow((total_lfp_modelfit), **imkw)
        axes2[1].imshow(lfp_sm_transf, **imkw)

        axes2[0].set_title("raw lfp (bandpass-filtered)")
        axes2[1].set_title("shift aligned, fit on lfp")

        axes2[0].set_ylabel("trials")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('LFP for f1815 cruella')

        sns.lineplot(ax=axes[0], x=lfp_time_crop, y=np.mean(total_lfp_modelfit[:,:,0], axis=0))
        sns.lineplot(ax=axes[1], x=lfp_time_crop, y=np.mean(lfp_sm_transf, axis=0))
        axes[0].set_title("raw lfp (bandpass-filtered)")
        axes[1].set_title("shift aligned-- fit on lfp")


        axes[0].set_ylabel("a.u.")
        plt.show()




