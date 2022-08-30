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
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import cwt, find_peaks_cwt

from scipy.signal import butter, lfilter
from func_spikes_targetonset import *

f={}
blockData={}

left_hand_or_right=['BB2BB3']
#ferret ID:
fid='F1702_Zola_Nellie'
fid='F1815_Cruella'
pitch_shift_or_not=['correctresp']
for k00 in pitch_shift_or_not:
   # blocksOfInterest = list(range(155,165))
    blocksOfInterest = [92,93,94,95,96,97,98,99,100,101,102]
    blocksOfInterest=list(range(92, 123))


    blocksOfInterest2 = []
    f = {}
    blockData = {}
    for k0 in left_hand_or_right:
        for i in blocksOfInterest:
            user_input = 'D:/Electrophysiological_Data/'+fid+'/LFP_BlockNellie-' + str(
                i) + '/noepoch/' + '/' + k0 + '/'

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


        TMIN = 0 * 1000  # s


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

        L = dict(np.load("umi_lfp_data.npz")) #for checking example data structure
        #
        # # Define bandpass filtering function for LFP
        import scipy.signal as signal
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

        def bandpass_low(x, lowcut, fs, order=5, axis=-1, kind='butter'):
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

            if kind == "butter":
                sos = butter(6, lowcut, 'low', fs=fs, output='sos')
            else:
                raise ValueError("Filter kind not recognized.")
            return signal.sosfilt(sos, x)




        # Fit to binned spike times.
        [shift_model, lin_model, data2, cropped_data2]=disgustingly_long_func2(pitch_shift_or_not, left_hand_or_right, blocksOfInterest2, fid)
        total_lfp_np=(np.array(total_lfp))

        fs=np.round(24414.0625*(1000/24414))

        ##making copy, z-scoring, then getting model fit
        # total_lfp_modelfit=(np.array(total_lfp))
        # #repeating S6 word for word, so I need to take the mean across channels and then add an extra dimension to make one "unit" for the spikedata object.
        #
        #
        def lowpass_by_site(data, lowcut):
            for k in range(0, 32):
                print(k)
                chosensite=data[:,:, k]
                corresp_bp=bandpass_low(chosensite,  lowcut, fs)
                data[:,:, k]=corresp_bp
            return data

        def bandpass_by_site(data, lowcut, highcut, fs):
            for k in range(0, 32):
                print(k)
                chosensite=data[:,:, k]
                corresp_bp=bandpass(chosensite,  lowcut, highcut, fs)
                data[:,:, k]=corresp_bp
            return data


        # NOTE TO FUTURE SELF, CHANGE LOW PASS AND HIGH PASS HERE:
        # 5,20
        # in future need to make function to loop this over different bands, e.g. 5-20, 5-30
        # need to check 15,20 again

        #total_lfp_np=bandpass_by_site(total_lfp_np, 1*1000, 20*1000, 1000)
        #total_lfp_np=np.mean(total_lfp_np, axis=2)
        # total_lfp_modelfit=total_lfp_modelfit[:,:, np.newaxis]
        start = 0
        stoptime = 5
        #fs=np.round(24414.0625*(1000/24414))
        lfp_time = np.linspace(start*fs*1000, stoptime*fs*1000, num=int(fs * (stoptime - start)+1))
        start_crop = 0*1000
        start_crop_lfp=0*1000
        stoptime_crop = 0.8*1000
        stoptime_crop_lfp=5*1000

        tidx = (lfp_time>= start_crop*fs) & (lfp_time <= stoptime_crop*fs)

        tidx_lfp=(lfp_time>= start_crop_lfp*fs) & (lfp_time <= stoptime_crop_lfp*fs)
        total_lfp_np = total_lfp_np[:, tidx_lfp, :]


        lfp_time_crop = lfp_time[tidx_lfp]
        stddevcalc= total_lfp_np.std(axis=(1), keepdims=True)

        total_lfp_np /= total_lfp_np.std(axis=1, keepdims=True)

        shift_model_lfp=shift_model.transform(total_lfp_np)
        lin_model_lfp=lin_model.transform(total_lfp_np)[:, :, 0]

        total_lfp_for_mod=total_lfp_np


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
        shift_model_on_lfp.fit(total_lfp_for_mod, iterations=3)
        lfp_sm_transf=shift_model_on_lfp.transform(total_lfp_for_mod)[:, :, 0]

        lin_model_on_lfp.fit(total_lfp_for_mod, iterations=3)


        import numpy as np
        import matplotlib.pyplot as plt

        imkw = dict(clim=(-2, 2), cmap='bwr', interpolation="none", aspect="auto")

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 3.5))

        axes[0].imshow(np.mean(total_lfp_for_mod, axis=2), **imkw)
        axes[1].imshow(np.mean(total_lfp_for_mod, axis=2), **imkw)
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

        N = 5001
        # sample spacing
        T = 1.0 / 200.0
        x = np.linspace(0.0, N * T, N, endpoint=False)

        y = np.mean(np.mean(total_lfp_np, axis=2), axis=0)
        y=scipy.signal.detrend((y))
        yf = fft(y)
        xf = fftfreq(N, T)[:N // 2]

        from scipy.signal import blackman

        w = blackman(N)
        ywf = fft(y * w)
        xf = fftfreq(N, T)[:N // 2]

        plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), label='de-trended only')
        plt.plot(xf, 2.0 / N * np.abs(ywf[0:N // 2]), label='detrended, with blackman window')

        plt.grid()
        plt.title('FFT of LFP, trial averaged, de-trended')
        plt.xlabel('Frequency')
        plt.xlim((0,20))
        plt.xticks(np.arange(0, 20, step=1), rotation=45)  # Set label locations.
        plt.legend()

        plt.show()


        def fft_with_window(signalx, N, T):
            signal_fft_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2]), dtype="complex_")
            for i in range(0, (signalx).shape[2]):
                selected_unit = signalx[:, :, i]
                #print(selected_unit.shape)

                for ii in range(0, signalx.shape[0]):
                    selected_trial_ofunit = selected_unit[ii, :]
                    #print(selected_trial_ofunit.shape)
                    y = selected_trial_ofunit
                    #y = y - np.mean(y)

                    w = blackman(N)
                    ywf = fft(y * w)
                    xf = fftfreq(N, T)[:N // 2]
                    signal_fft_grid[ii, :, i] = ywf

            return signal_fft_grid


        from wavelets import WaveletAnalysis
        import pycwt
        from pycwt import wavelet

        def wavelet_kpanalysis(signalx, dt=0.25):
            signal_waveletpower_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 46),dtype="float")
            signal_waveletscales_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 46),dtype="float")
            signal_wavelettime_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 46),dtype="float")
            signal_waveletsignif_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 46),dtype="float")
            signal_waveletperiod_grid=np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 46),dtype="float")

            for i in range(0, (signalx).shape[2]):
                selected_unit = signalx[:, :, i]

                for ii in range(0, signalx.shape[0]):
                    selected_trial_ofunit = selected_unit[ii, :]
                    #print(selected_trial_ofunit.shape)
                    y = selected_trial_ofunit
                    slevel = 0.95  # Significance level

                    std = y.std()  # Standard deviation
                    std2 = std ** 2  # Variance
                    var = (y - y.mean()) / std  # Calculating anomaly and normalizing

                    dj = 0.25  # Four sub-octaves per octaves
                    s0 = -1  # 2 * dt                      # Starting scale, here 6 months
                    J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves

                    mother = wavelet.Morlet(6.)  # Morlet mother wavelet with wavenumber=6

                    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, dt, dj, s0, J,
                                                                          mother)

                    iwave = wavelet.icwt(wave, scales, dt, dj, mother)
                    power = (abs(wave)) ** 2  # Normalized wavelet power spectrum
                    fft_power = std2 * abs(fft) ** 2  # FFT power spectrum
                    period = 1. / freqs
                    #alpha=0.05
                    N = y.size;
                    #print(N)

                    #alpha, _, _ = wavelet.ar1(y)
                    alpha=np.corrcoef(y[:-1], y[1:])[0, 1]

                    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                                              significance_level=slevel, wavelet=mother)
                    sig95 = (signif * np.ones((N, 1))).transpose()
                    sig95 = power / sig95  # Where ratio > 1, power is significant

                    # Calculates the global wavelet spectrum and determines its significance level.
                    glbl_power = std2 * power.mean(axis=1)
                    dof = N - scales  # Correction for padding at edges
                    glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                                                            significance_level=slevel, dof=dof, wavelet=mother)


                    signal_waveletpower_grid[ii, :, i, :] =np.transpose(power)
                    signal_waveletscales_grid =scales
                    signal_waveletsignif_grid[ii, :, i, :]=np.transpose(sig95)

                    signal_waveletperiod_grid[ii, :, i, :]=np.transpose(period)

                    # associated time vector

                    signal_wavelettime_grid = np.arange(0, N) * dt

            return signal_waveletpower_grid, signal_wavelettime_grid, signal_waveletscales_grid, period, signal_waveletsignif_grid, signal_waveletperiod_grid

        def wavelet_wvanalysis(signalx, dt=5):
            signal_waveletpower_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 91),dtype="float")
            signal_waveletscales_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 91),dtype="float")
            signal_wavelettime_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 91),dtype="float")

            for i in range(0, (signalx).shape[2]):
                selected_unit = signalx[:, :, i]

                for ii in range(0, signalx.shape[0]):
                    selected_trial_ofunit = selected_unit[ii, :]
                    #print(selected_trial_ofunit.shape)
                    y = selected_trial_ofunit
                    wa = WaveletAnalysis(y, dt=800)
                    signal_waveletpower_grid[ii, :, i, :] = np.transpose(wa.wavelet_power)
                    signal_waveletscales_grid = wa.scales

                    # associated time vector
                    signal_wavelettime_grid = wa.time

            return signal_waveletpower_grid, signal_wavelettime_grid, signal_waveletscales_grid


        def wavelet_with_window(signalx, N, T):
            signal_wavelet_grid = np.empty(((signalx).shape[0], (signalx).shape[1], (signalx).shape[2], 50),dtype="float")
            #print(signal_wavelet_grid.shape())
            for i in range(0, (signalx).shape[2]):
                selected_unit = signalx[:, :, i]
                #print(selected_unit.shape)

                for ii in range(0, signalx.shape[0]):
                    selected_trial_ofunit = selected_unit[ii, :]
                    #print(selected_trial_ofunit.shape)
                    y = selected_trial_ofunit
                    widths = np.arange(1, 51)
                    cwtmatr = signal.cwt(y, signal.morlet2, widths)
                    signal_wavelet_grid[ii, :, i, :] = np.transpose(cwtmatr)

            return signal_wavelet_grid


        # sample spacing
        T = 1.0 / 200.0
        shift_model_lfp_forfft = scipy.signal.detrend((shift_model_lfp), axis=1)
        total_lfp_np_cwt=scipy.signal.detrend(total_lfp_np, axis=1)
        signal_in_grid = fft_with_window(shift_model_lfp_forfft, N, T)
        x = np.linspace(0.0, N * T, N, endpoint=False)
        y = np.mean(np.mean(signal_in_grid, axis=2), axis=0)

        signal_wavelet=wavelet_with_window(shift_model_lfp_forfft, N, T)
        signal_waveletpower_grid, signal_wavelettime_grid, signal_waveletscales_grid =wavelet_wvanalysis(shift_model_lfp_forfft)
        signal_waveletpower_grid2=np.mean(signal_waveletpower_grid, axis=0)
        signal_waveletpower_grid3=np.mean(signal_waveletpower_grid2, axis=1)
        fig, ax = plt.subplots()


        T, S = np.meshgrid(signal_wavelettime_grid, signal_waveletscales_grid)
        ax.contourf(T, S, np.transpose(signal_waveletpower_grid3), 100)
        #ax.set_yticks(np.linspace(0, 100, 10))
        #ax.set_yscale('log')
        plt.show()


        signal_waveletpower_grid, signal_wavelettime_grid, signal_waveletscales_grid, period, signal_waveletsignif_grid, signal_waveletperiod_grid =wavelet_kpanalysis(total_lfp_np_cwt)
        signal_waveletpower_grid4=np.mean(signal_waveletpower_grid, axis=0)
        signal_waveletpower_grid5=np.mean(signal_waveletpower_grid4, axis=1)

        signal_waveletsignif_grid2=np.mean(signal_waveletsignif_grid, axis=0)
        signal_waveletsignif_grid3=np.mean(signal_waveletsignif_grid2, axis=1)
        signal_waveletperiod_grid2=np.mean(signal_waveletperiod_grid, axis=0)
        signal_waveletperiod_grid3 = np.mean(signal_waveletperiod_grid2, axis=1)
        signal_waveletperiod_grid4=np.mean(signal_waveletperiod_grid3, axis=0)


        f, ax = plt.subplots(figsize=(15, 10))
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        levels=20
        vmin=-30
        vmax=10

        cruellacwt=ax.contourf(signal_wavelettime_grid*4, np.log2(signal_waveletperiod_grid4), np.log2(np.transpose(signal_waveletpower_grid5)),levels, vmin=vmin, vmax=vmax,
                    extend='both')
        cruellacwtlines=ax.contour(signal_wavelettime_grid*4, np.log2((signal_waveletperiod_grid4)), np.transpose(signal_waveletsignif_grid3), [-99, 1], colors='k',
                   linewidths=2.)
        ax.set_yticks(np.arange(0,10,1))


        ax.set_xticks(np.arange(0,5000, 500))

        cbar = f.colorbar(cruellacwt)
        cbar.add_lines(cruellacwtlines)
        cbar.ax.set_ylabel('power')

        plt.xlabel('Time (ms), trial onset = 0.5s, 500ms')
        plt.ylabel('frequency')
        plt.title('kPywavelet version - Cruella lfp, original lfp, 3s<time to target word<4s')
        plt.show()


        #
        # cwtmatr0=np.mean(signal_wavelet, axis=0)
        # cwtmatr2=np.mean(cwtmatr0, axis=1)
        # cwtmatr2=np.transpose(cwtmatr2)
        # plt.imshow(cwtmatr2, extent=[-1, 1, 51, 1], cmap='PRGn', aspect='auto',
        #            vmax=abs(cwtmatr2).max(), vmin=-abs(cwtmatr2).max())
        # plt.show()
        # plt.title('trial-averaged global LFP, wavelet transform morlet2')



        # y = np.mean(signal_in_grid[:,:,2],axis=0)

        # y=scipy.signal.detrend((y), type='constant')

        # yf = fft(y)
        # xf = fftfreq(N, T)[:N // 2]
        #
        # w = blackman(N)
        # ywf = fft(y * w)
        # xf = fftfreq(N, T)[:N // 2]

        # f, Pxx_den = signal.welch(y, fs, nperseg=1024)
        # plt.semilogy(f, Pxx_den)
        # plt.ylim([0.5e-3, 1])
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('PSD [V**2/Hz]')
        # plt.show()

        plt.plot(xf, 2.0 / N * np.abs(y[0:N // 2]), label='shift model, 5s duration')

        plt.grid()
        plt.title('FFT blackman window then fft taken per unit per trial, then trial averaged, de-trended', fontsize=9)
        plt.xlabel('Frequency')
        plt.xlim((0, 20))
        plt.xticks(np.arange(0, 20, step=1), rotation=45)  # Set label locations.
        plt.legend()

        plt.show()


        # sample spacing
        T = 1.0 / 200.0
        lfp_forfft = scipy.signal.detrend((total_lfp_np), axis=1)
        signal_in_grid2 = fft_with_window(lfp_forfft, N, T)
        x = np.linspace(0.0, N * T, N, endpoint=False)
        y = np.mean(np.mean(signal_in_grid2, axis=2), axis=0)

        plt.plot(xf, 2.0 / N * np.abs(y[0:N // 2]), label='LFP, 5s duration')

        plt.grid()
        plt.title(
            'FFT of LFP, blackman window then fft taken per unit per trial, then trial averaged, de-trended', fontsize=9)
        plt.xlabel('Frequency')
        plt.xlim((0, 20))
        plt.xticks(np.arange(0, 20, step=1), rotation=45)  # Set label locations.
        plt.legend()

        plt.show()



        fig2, axes2 = plt.subplots(1, 3, sharey=True, figsize=(10, 3.5))
        lfp_np_plt=np.mean(np.mean(total_lfp_np, axis=2), axis=0)
        shift_model_lfp_plt=np.mean(np.mean(shift_model_lfp, axis=2), axis=0)
        lin_model_lfp_plt=np.mean(lin_model_lfp, axis=0)


        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle('LFP for'+fid)

        sns.lineplot(ax=axes[0], x=lfp_time_crop, y=lfp_np_plt)
        sns.lineplot(ax=axes[1], x=lfp_time_crop, y=shift_model_lfp_plt)
        sns.lineplot(ax=axes[2], x=lfp_time_crop, y=lin_model_lfp_plt)
        axes[0].set_title("raw, normalised lfp (bandpass-filtered) "+ fid)
        axes[1].set_title("shift aligned on normalised LFP")
        axes[2].set_title("linear aligned on normalised LFP")

        axes[0].set_ylabel("a.u.")
        axes[0].set_xlabel("raw time values")

        plt.show()



        fractional_shifts_spk_warp=shift_model.fractional_shifts

        # fractional_shifts_lfp_warp_idx=shift_model_on_lfp.fractional_shifts>=0
        # fractional_shifts_lfp_warp=shift_model_on_lfp.fractional_shifts[fractional_shifts_spk_warp_idx]
        fractional_shifts_lfp_warp=shift_model_on_lfp.fractional_shifts


        fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
        corrcoeff=scipy.stats.pearsonr(fractional_shifts_spk_warp, fractional_shifts_lfp_warp)
        sns.scatterplot(x=fractional_shifts_spk_warp, y=fractional_shifts_lfp_warp)
        plt.title('fractional shifts of spike model vs. lfp model, '+fid)
        plt.xlabel('Spike model fractional shifts')
        plt.ylabel('LFP model fractional shifts')
        plt.show()

        
        fig2, axes2 = plt.subplots(1, 2, sharey=True, figsize=(10, 3.5))
        axes2[0].imshow(np.mean(total_lfp_for_mod, axis=2), **imkw)
        axes2[1].imshow(lfp_sm_transf, **imkw)

        axes2[0].set_title("raw lfp "+fid)
        axes2[1].set_title("shift aligned, fit on lfp")

        axes2[0].set_ylabel("trials")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('LFP for f1815 cruella')
        plttest=total_lfp_for_mod[:,:,0]
        sns.lineplot(ax=axes[0], x=lfp_time_crop, y=lfp_np_plt)
        sns.lineplot(ax=axes[1], x=lfp_time_crop, y=np.mean(lfp_sm_transf, axis=0))
        axes[0].set_title("raw lfp (trial-averaged) "+fid)
        axes[1].set_title("shift aligned-- fit on lfp " + fid)


        axes[0].set_ylabel("mV")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        sns.lineplot(ax=axes[0], x=lfp_time_crop, y=np.mean(total_lfp_for_mod[:, :, 3], axis=0))
        sns.lineplot(ax=axes[1], x=lfp_time_crop, y=lfp_np_plt)
        plt.show()


        def crosscorr(datax, datay, lag=0, wrap=False):
            """ Lag-N cross correlation.
            Shifted data filled with NaNs

            Parameters
            ----------
            lag : int, default 0
            datax, datay : pandas.Series objects of equal length
            Returns
            ----------
            crosscorr : float
            """
            if wrap:
                print(lag)
                shiftedy = datay.shift(lag)
                shiftedy.iloc[:lag] = datay.iloc[-lag:].values
                print(shiftedy)
                return datax.corr(shiftedy)
            else:

                print('data y shift:')
                # datax['newy']=datay.shift(lag)
                # print(datax.corr())
                # datax.corrwith(datay.shift(lag), axis=0)
                print(datay.shift(lag))

                return datax.corrwith(datay.shift(lag), axis=0)


        d1 = (cropped_data2.spiketimes)
        [hist_d1, bin_edges]=np.histogram(d1, bins=801, range=None, normed=None, weights=None, density=None)
        d1=pd.DataFrame(hist_d1)

        ##need to do hiscounts of spike times across 800 ms with number of bins
        d2 =pd.DataFrame(np.mean(np.mean(total_lfp_np, axis=2), axis=0))
        plt.plot(hist_d1, label='hist counts of spikes across all 32 sites')
        plt.plot(d2, label ='averaged LFP across all trials, across all 32 sites ')
        plt.show()
        seconds = 0.8
        fps = 1000


        window = 10
        # lags = np.arange(-(fs), (fs), 1)  # uncontrained
        #make half or less of the timeseries shifted array have resulting NAN values or else this will return NONSENSICAL results

        lags = np.arange(-(200), (200), 1)  # contrained
        rs = np.nan_to_num([crosscorr(d1, d2, lag) for lag in lags])

        print(
            "xcorr {}-{}".format(d1, d2, lags[np.argmax(rs)], np.max(rs)))

        #
        # rs = [crosscorr(d1, d2, lag) for lag in range(0, int(seconds * fps + 1))]
        offset = np.floor(len(rs) / 2) - np.argmax(rs)
        f, ax = plt.subplots(figsize=(14, 3))
        ax.plot(rs)
        ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
        ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
        ax.set(title=f'Offset = {offset} frames\n Spike activity leads <> LFP activity leads', xlabel='Offset',
               ylabel='Pearson r')
        # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
        # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
        plt.legend()
        plt.show()





