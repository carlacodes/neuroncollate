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
base_path = 'D:\Data\Deepinsight Zola'
f = h5py.File('D:\Data\models\Deepinsight ZolaDataBlockNellie-28_model_4.h5', 'r')
fp_deepinsight = base_path + 'DataBlockNellie-28.h5'  # This will be the processed HDF5 file
hdf5_file = h5py.File(fp_deepinsight, mode='r')
analysisvar = hdf5_file["analysis"]
analysisvar = np.transpose(analysisvar)
analysisvar=np.array(analysisvar)
predictions = hdf5_file["analysis/predictions"]

losses = hdf5_file["analysis/losses"][()]
shuffled_losses = hdf5_file["analysis/influence/shuffled_losses"][()]

f2 = h5py.File('D:\Electrophysiological Data\stretchedstimuliMarch-25-2020-10-47-00-062-AM.mat', 'r')
variables = f.items()
print(variables)

for var in variables:
    name = var[0]
    data = var[1]
    print ("Name ", name )# Name
    if type(data) is h5py.Dataset:
        print('hello')
        print(data[()])
# If DataSet pull the associated Data
# If not a dataset, you may need to access the element sub-items
