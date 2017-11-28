# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import time

from jingjuElemSeg.src import noteClass as nc
from jingjuElemSeg.src import trainTestKNN as ttknn
from jingjuElemSeg.src import refinedSegmentsManipulation as rsm


def jingjuElementSegmentation(workdir):

    start_time = time.time()  # starting time
    '''
    ######################################### pYIN pitchtrack and notes ################################################
    filename_amateur = '/Users/gong/Documents/MTG document/Jingju arias/jingjuElementMaterials/laosheng/test/weiguojia_section_amateur.wav'
    filename_pro = '/Users/gong/Documents/MTG document/Jingju arias/jingjuElementMaterials/laosheng/test/weiguojia_section_pro.wav'
    pYINPtNote(filename_amateur)
    #filename1 = pYinPath + '/testAudioLong.wav'
    '''

    ############################################## initialsation #######################################################

    # classification training ground truth
    # groundtruthNoteLevelPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/pyinNoteCurvefit/classified'
    # groundtruthNoteDetailPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/refinedSegmentCurvefit/classified'

    # classification model path
    pitchContourClassificationModelName = '/data/pYinOut/laosheng/train/model/pitchContourClassificationModel.pkl'

    # feature, target folders train
    # featureVecTrainFolderPath = os.path.join(dir,'pYinOut/laosheng/train/featureVec/')
    # targetTrainFolderPath = os.path.join(dir,'pYinOut/laosheng/train/target/')

    # predict folders
    pitchtrackNotePredictFolderPath = workdir
    featureVecPredictFolderPath = os.path.join(workdir, 'predict/featureVec/')
    targetPredictFolderPath = os.path.join(workdir, 'predict/target/')
    try:
        os.makedirs(featureVecPredictFolderPath)
    except OSError:
        pass
    try:
        os.makedirs(targetPredictFolderPath)
    except OSError:
        pass

    # recordingNamesTrain = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    recordingNamesPredict = ['student']

    slopeTh = 60.0                          #  contour combination slope difference threshold
    flatnoteTh = 80.0                       #  threshold for deciding one note as flat pitch note
    #recordingNamesPredict = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']       # evaluation

    ################################################## predict #########################################################
    # segmentation
    nc2 = nc.noteClass()
    nc2.noteSegmentationFeatureExtraction(pitchtrackNotePredictFolderPath,
                                          featureVecPredictFolderPath,
                                          pitchtrackNotePredictFolderPath,
                                          recordingNamesPredict,
                                          segCoef=0.3137, predict=True)

    # predict
    ttknn2 = ttknn.TrainTestKNN()
    ttknn2.predict(pitchContourClassificationModelName, featureVecPredictFolderPath,
                   targetPredictFolderPath, recordingNamesPredict)


    ########################################### representation #########################################################
    for rm in recordingNamesPredict:
        #  filename declaration
        originalPitchtrackFilename = os.path.join(pitchtrackNotePredictFolderPath, rm+'_pitchtrack.csv')
        targetFilename = os.path.join(targetPredictFolderPath, rm+'.json')
        refinedSegmentFeaturesFilename = os.path.join(pitchtrackNotePredictFolderPath, rm+'_refinedSegmentFeatures.json')
        representationFilename = os.path.join(pitchtrackNotePredictFolderPath, rm+'_representation.json')
        figureFilename = os.path.join(pitchtrackNotePredictFolderPath, rm+'_reprensentationContourFigure.png')

        # important txt files!!
        regressionPitchtrackFilename = os.path.join(pitchtrackNotePredictFolderPath, rm+'_regression_pitchtrack.csv')
        refinedSegmentationGroundtruthFilename = os.path.join(pitchtrackNotePredictFolderPath, rm+'_refinedSeg.csv')

        rsm1 = rsm.RefinedSegmentsManipulation()
        rsm1.process(refinedSegmentFeaturesFilename, targetFilename,
                     representationFilename, figureFilename, regressionPitchtrackFilename,
                     originalPitchtrackFilename=originalPitchtrackFilename,
                     refinedSegGroundtruthFilename=refinedSegmentationGroundtruthFilename,
                     slopeTh=slopeTh, flatnoteTh=flatnoteTh)

    runningTime = time.time() - start_time
    # print("--- %s seconds ---" % (time.time() - start_time))

    return runningTime
