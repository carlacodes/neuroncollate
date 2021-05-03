import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
from scipy.io import loadmat
import os
from glob import glob
dpath = 'D:/crcns/ofc-2/data/'

rats = os.listdir(dpath)
sessions = {r: os.listdir(os.path.join(dpath, r)) for r in rats}
tmax = 6.0  # maximum time for a trial


def load(rat=0, sess=0):
    r = rats[rat] if type(rat) == int else rat
    p = os.path.join(dpath, r, sessions[r][sess])
    print("Loading Session: " + p)
    # spike times in ms
    spiketimes = [loadmat(f)['TS'].ravel() for f in glob(os.path.join(p, 'Sc*'))]

    ev1 = loadmat(os.path.join(p, 'TrialEvents.mat'))
    ev2 = loadmat(os.path.join(p, 'TrialEvents2.mat'))

    # collect the important metadata
    info = {
        'correct': ev2['Correct'].ravel(),
        'stim': ev2['OdorCategory'].ravel(),
        'choice': ev2['ChoiceDir'].ravel(),
        'ratio': ev2['OdorRatio'].ravel(),
        'odor_in': ev1['TrialStart'].ravel() + ev1['OdorPokeIn'].ravel(),
        'odor_out': ev1['TrialStart'].ravel() + ev1['OdorPokeOut'].ravel(),
        'water_in': ev1['TrialStart'].ravel() + ev1['WaterPokeIn'].ravel(),
        'water_on': ev1['TrialStart'].ravel() + ev2['WaterValveOn'].ravel(),
        'water_out': ev1['TrialStart'].ravel() + ev1['WaterPokeOut'].ravel(),
    }

    # filter out incorrect trials
    valid = np.isfinite(info['water_on']) & np.isfinite(info['odor_in'])
    info = {k: v[valid] for k, v in info.items()}

    # align valid trials relative to odor_in
    refpt = info['odor_in'].copy() * 10000
    for k in ('odor_in', 'odor_out', 'water_in', 'water_on', 'water_out'):
        info[k] = (10000 * info[k] - refpt).astype(int)

    # get trial indices for each spike
    trials, times, neurons = [], [], []
    for n, st in enumerate(spiketimes):
        trials += get_trial_idx(st, refpt)
        times += st.astype(int).tolist()
        neurons += np.full(len(st), n).tolist()

    # convert to numpy arrays
    trials, times, neurons = np.array(trials), np.array(times), np.array(neurons)

    # filter out any spikes that weren't matched to a trial and spikes over 2000 bins after trial start
    idx = (trials >= 0) & (times < 60000)

    return trials[idx], times[idx], neurons[idx], info


def get_trial_idx(st, refpt):
    K = len(refpt)
    trials = np.full(len(st), -1)
    _get_trials(trials, st, refpt)
    return trials.tolist()


def _get_trials(trials, times, start_times):
    # trial counter
    K = len(start_times)
    k = 0

    # add each spike to M
    for i in range(len(times)):

        # ignore spikes preceding first trial
        if times[i] < start_times[0]:
            continue

        # advance to next trial
        while (k + 1 < K) and (times[i] > start_times[k + 1]):
            k += 1

        # record trial
        t = times[i] - start_times[k]

        if t > 0:
            trials[i] = k
            times[i] = t


RAT = 0
SESS = 3

# Load the data.
fields = ['trials', 'times', 'neurons', 'metadata']
trials, times, neurons, metadata = load(rat=RAT, sess=SESS)
np.savez('ofc2_data.npz', trials=trials, neurons=neurons, times=times)
np.savez('ofc2_metadata.npz', **metadata)