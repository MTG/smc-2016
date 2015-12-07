import sys,os

# the project folder: fileDir
# fileDir = os.path.dirname(os.path.realpath('__file__'))
# dtwPath = os.path.join(fileDir, '../Library_PythonNew/similarityMeasures/dtw/')
# transcribe the path string to full path
# dtwPath = os.path.abspath(os.path.realpath(dtwPath))

# sys.path.append(dtwPath)

import dtw
import numpy as np

def dtw1d_generic(x, y):

    configuration = {}
        # myDtwParama.distType = 0;
#     myDtwParama.hasGlobalConst = 1;
#     myDtwParama.globalType = 0;
#     myDtwParama.bandwidth = np.round(x_arr.shape[0]*0.2).astype(np.int);
#     myDtwParama.initCostMtx = 1;
#     myDtwParama.reuseCostMtx = 0;
#     myDtwParama.delStep = 1;
#     myDtwParama.moveStep = 1;
#     myDtwParama.diagStep = 1;
#     myDtwParama.initFirstCol = 1;
#     myDtwParama.isSubsequence = 1;
    configuration['distType'] = 1                   # square euclidean
    configuration['hasGlobalConst'] = 1
    configuration['globalType'] = 0
    configuration['bandwidth'] = 0.2
    configuration['initCostMtx'] = 1
    configuration['reuseCostMtx'] = 0
    configuration['delStep'] = 1
    configuration['moveStep'] = 1
    configuration['diagStep'] = 1
    configuration['initFirstCol'] = 1
    configuration['isSubsequence'] = 1

    udist, plen, path, cost_arr  = dtw.dtw1d_GLS(x, y, configuration)

    return path


