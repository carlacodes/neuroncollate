# load data
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
ofc2 = np.load('ofc2_data.npz')

# spike times
trials = ofc2['trials']
times = ofc2['times']
neurons = ofc2['neurons']

# dict holding trials specific data
#  --> e.g., metadata['water_on'] gives the time that the reward water turned on each trial
metadata = np.load('ofc2_metadata.npz')
from affinewarp import SpikeData

data = SpikeData(trials, times, neurons, tmin=0.0, tmax=6e4)

n_bins = 100
binned = data.bin_spikes(n_bins)
n_trials = binned.shape[0]

from affinewarp.visualization import rasters

fig, axes = rasters(data, subplots=(4, 4), s=2);

# Plot the reward time for each trial.
# for ax in axes.ravel()[:-1]:
#     ax.scatter(metadata['water_on'], np.arange(data.n_trials), c='r', s=.5, alpha=.5)
plt.show()
from affinewarp import PiecewiseWarping
model = PiecewiseWarping(n_knots=0)

# fit model and show loss history
model.fit(binned, iterations=20, warp_iterations=20)
plt.plot(model.loss_hist)
plt.xlabel('iterations')
plt.ylabel('reconstruction error');

# warp each spike
warped_spikes = model.transform(data)
plt.show()

# warp the onset of reward for each trial
warped_water = 6e4 * model.event_transform(np.arange(data.n_trials), metadata["water_on"] / 6e4)

# plot rasters
fig, axes = rasters(warped_spikes, subplots=(4, 4), s=2);
plt.show()

# plot water onset
K = len(binned)
for ax in axes.ravel()[:-1]:
    ax.scatter(warped_water, np.arange(data.n_trials), c='r', s=.5, alpha=.5)


plt.show()
