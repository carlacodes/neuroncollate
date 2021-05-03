# import tensorflow as tf
import deepinsight
# Choose GPU
import os
# import tensorflow as tf
import scipy
import tensorflow as tf
import h5py
import numpy as np

# tf.device("/GPU:0")
##end of selecting packaegs to analyse

print(tf.test.is_gpu_available())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
y = 4
print(y)
# response = urllib.request.urlopen("https://ndownloader.figshare.com/files/20150468/")
##March 2, 2020 test of the deepinsight network
base_path = 'D:/Deepinsight Zola/'
f = h5py.File('D:\Electrophysiological Data\deepinsightDataBlockNellie-28March-24-2020- 4-50-57-498-PM.mat', 'r')

f2 = h5py.File('D:\Electrophysiological Data\stretchedstimuliMarch-25-2020-10-47-00-062-AM.mat', 'r')
variables = f.items()
print(variables)

# for var in variables:
#     name = var[0]
#     data = var[1]
#     print ("Name ", name )# Name
#     if type(data) is h5py.Dataset:
#         print(name)
# If DataSet pull the associated Data
# If not a dataset, you may need to access the element sub-items
# value = data.value
# print("Value", value)  # NumPy Array / Value

data = f.get('kiloDataAllChannels')
timestampdata = f.get('raw_timestamps')
fp_deepinsight = base_path + 'DataBlockNellie-28.h5'  # This will be the processed HDF5 file
raw_data = np.transpose(data)
raw_data = np.array(raw_data)

raw_timestamps = np.array(timestampdata)
#raw_timestamps=np.array(raw_timestamps)
print(raw_data.shape)
print(raw_timestamps.shape)

output=f2.get('stretchedstimuli')
output=np.transpose(output)
output=np.array(output)
output_timestamps=f2.get('stretchedtimestamp')
output_timestamps=np.transpose(output_timestamps)
output_timestamps=np.array(output_timestamps)


print('output dimensions', output.shape)
print('output dimensions', output_timestamps.shape)

#raw_timestamps=raw_timestamps[:,0]
#output_timestamps=output_timestamps[:,0]
#outputPatternlength = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0], average_window)], output_timestamps[:,0], output[2, :])

#output=output[:,0]

# np.ndarray.flatten(output_timestamps)
# np.ndarray.flatten(output)

# deepinsight.preprocess.preprocess_input(fp_deepinsight, raw_data, sampling_rate=24414.06250000000, channels=range(1,64))
deepinsight.util.tetrode.preprocess_output(fp_deepinsight, raw_timestamps, output, output_timestamps, sampling_rate=24414.06250000000)

# loss_functions = {'target_word' : 'mae'}
# loss_weights = {'target_word' : 1}

# fp_raw_file = base_path + 'experiment_1.nwb' # This is your raw file

#
#
loss_functions = {'words' : 'mae',
                  'timestamps' : 'mae'}
loss_weights = {'words' : 2,
                  'timestamps' :2}
deepinsight.train.run_from_path(fp_deepinsight, loss_functions, loss_weights)

losses, output_predictions, indices = deepinsight.analyse.get_model_loss(fp_deepinsight,
                                                                         stepsize=10)
shuffled_losses = deepinsight.analyse.get_shuffled_model_loss(fp_deepinsight, axis=1,
                                                              stepsize=10)


#Plot influence across behaviours
deepinsight.visualize.plot_residuals(fp_deepinsight, frequency_spacing=2,
                                     output_names=['Target Word', 'Timestamp'])
