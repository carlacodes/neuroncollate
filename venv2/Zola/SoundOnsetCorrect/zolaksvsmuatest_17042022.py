##using july 12 2021 data, correct responses -0.3s to 0.3s relative to trial onset and offset
import numpy as np
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping
import os
import h5py
import numpy as np
import pandas as pd
k0='BB2BB3'
BASE_PATH2 = 'D:/Electrophysiological Data/F1702_Zola/dynamictimewarping/l27soundonset/' + k0 + '/'
file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdata.NPY'
cropped_data2=np.load(os.path.join(BASE_PATH2, file_name), allow_pickle=True)
# with (open(os.path.join(BASE_PATH2, file_name), "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

object = pd.read_pickle(os.path.join(BASE_PATH2, file_name))

# file_name = 'alignedDataBlockweekjuly122021OriginalModelrasterdataNbins'
# np.save(os.path.join(BASE_PATH2, file_name), NBINS)
# # TMIN, TMAX, sorted_array, combinedTrials, epoch_offset
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