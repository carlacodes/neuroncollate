
import numpy as np
import matplotlib.pyplot as plt
import affinewarp
import scipy.io as sio
from os.path import dirname, join as pjoin
import h5py

data_dir='D:\Electrophysiological Data\F1702_Zola_Nellie\SpikeSorting29012021'
mat_fname = pjoin(data_dir, 'spikeswv39April-28-2021.h5')

#import scipy.optimize.nnls
#matplotlib inline
mat_contents =  h5py.File(mat_fname)
data1 = mat_contents.get('wv39testpos').value
data2=data1.transpose()

data2 = data2[..., np.newaxis]
list(mat_contents.keys())
struArray = mat_contents['wv39testpos']
# Helper function for generating shifted data
from affinewarp.datasets import jittered_data
#data = jittered_data(t=np.linspace(-10,10,150), jitter=2, noise=.2)[-1]
data=data2 #test 18/04/2021
# Plot data.
plt.imshow(np.squeeze(data), aspect='auto')
plt.title('raw data'), plt.xlabel('time (a.u.)'), plt.ylabel('trials')
plt.colorbar();
print('line finished')
plt.show()
from affinewarp import ShiftWarping

# Create the model. Add a roughness penalty to the model template.
model = ShiftWarping(maxlag=.3, smoothness_reg_scale=10.)

# NOTE : you can also use PiecewiseWarping with `n_knots` parameter set to -1.
#
#  >> model = PiecewiseWarping(n_knots=-1, smoothness_reg_scale=10.)

# Fit the model.
model.fit(data, iterations=20)

# Plot model learning curve.
plt.plot(model.loss_hist)
plt.xlabel('iterations')
plt.ylabel('model loss');
plt.show()

plt.imshow(model.predict().squeeze(), aspect='auto')
plt.title('denoised data'), plt.xlabel('time (a.u.)'), plt.ylabel('trials')
plt.colorbar();
plt.show()


plt.imshow(model.transform(data).squeeze(), aspect='auto')
plt.title('aligned raw data'), plt.xlabel('time (a.u.)'), plt.ylabel('trials')
plt.colorbar();
plt.show()


plt.imshow(np.tile(model.template, (1, data.shape[0])).T, aspect='auto')
plt.title('aligned denoised data'), plt.xlabel('time (a.u.)'), plt.ylabel('trials')
plt.colorbar();
plt.show()

plt.plot(model.transform(data)[:,:,0].mean(axis=0), color='k', label='warped average')
plt.plot(data.mean(axis=0), color='r', label='naive average')
plt.ylabel('trial-averaged neural activity'), plt.xlabel('time (a.u.)'), plt.legend();
plt.show()


plt.plot(model.transform(data)[:,:,0].mean(axis=0), color='k', label='warped average')
plt.plot(model.template, '-', color='b', label='model template', lw=3, alpha=.85)
plt.ylabel('neural activity'), plt.xlabel('time (a.u.)'), plt.legend();
plt.show()