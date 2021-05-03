#import tensorflow as tf
import deepinsight
# Choose GPU
import os
#import tensorflow as tf
import scipy
import tensorflow as tf
#tf.device("/GPU:0")
##end of selecting packaegs to analyse

print(tf.test.is_gpu_available())
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
y=4
print(y)
#response = urllib.request.urlopen("https://ndownloader.figshare.com/files/20150468/")
##March 2, 2020 test of the deepinsight network
base_path = 'D:/Data/DeepInsight Example/'
fp_raw_file = base_path + 'experiment_1.nwb' # This is your raw file
fp_deepinsight = base_path + 'processed_R2478.h5' # This will be the processed HDF5 file


loss_functions = {'position' : 'euclidean_loss',
                  'head_direction' : 'cyclical_mae_rad',
                  'speed' : 'mae'}
loss_weights = {'position' : 1,
                'head_direction' : 25,
                'speed' : 2}
deepinsight.train.run_from_path(fp_deepinsight, loss_functions, loss_weights)

losses, output_predictions, indices = deepinsight.analyse.get_model_loss(fp_deepinsight,
                                                                         stepsize=10)
shuffled_losses = deepinsight.analyse.get_shuffled_model_loss(fp_deepinsight, axis=1,
                                                              stepsize=10)


# Plot influence across behaviours
deepinsight.visualize.plot_residuals(fp_deepinsight, frequency_spacing=2,
                                     output_names=['Position', 'Head Direction', 'Speed'])