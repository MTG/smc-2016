# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import time

from jingjuElemSeg.src import noteClass as nc
from jingjuElemSeg.src import trainTestKNN as ttknn
from jingjuElemSeg.src import refinedSegmentsManipulation as rsm


def jingjuElementSegmentation():

    start_time = time.time()  # starting time
    '''
    ######################################### pYIN pitchtrack and notes ################################################
    filename_amateur = '/Users/gong/Documents/MTG document/Jingju arias/jingjuElementMaterials/laosheng/test/weiguojia_section_amateur.wav'
    filename_pro = '/Users/gong/Documents/MTG document/Jingju arias/jingjuElementMaterials/laosheng/test/weiguojia_section_pro.wav'
    pYINPtNote(filename_amateur)
    #filename1 = pYinPath + '/testAudioLong.wav'
    '''

    ############################################## initialsation #######################################################
    pitchtrackNoteTrainFolderPath = os.path.join(dir, 'pYinOut/laosheng/train/')

    # classification training ground truth
    # groundtruthNoteLevelPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/pyinNoteCurvefit/classified'
    # groundtruthNoteDetailPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/refinedSegmentCurvefit/classified'

    # classification model path
    pitchContourClassificationModelName = os.path.join(dir,'pYinOut/laosheng/train/model/pitchContourClassificationModel.pkl')

    # feature, target folders train
    # featureVecTrainFolderPath = os.path.join(dir,'pYinOut/laosheng/train/featureVec/')
    # targetTrainFolderPath = os.path.join(dir,'pYinOut/laosheng/train/target/')

    # predict folders
    pitchtrackNotePredictFolderPath = os.path.join(dir,'../../data/updateFiles/')
    #pitchtrackNotePredictFolderPath = os.path.join(dir,'pYinOut/laosheng/predict/)
    featureVecPredictFolderPath = os.path.join(dir,'pYinOut/laosheng/predict/featureVec/')
    targetPredictFolderPath = os.path.join(dir,'pYinOut/laosheng/predict/target/')


    # recordingNamesTrain = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    recordingNamesPredict = ['student']

    evaluation = False                      #  parameters grid search

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
                                          segCoef=0.3137,predict=True)

    # predict
    ttknn2 = ttknn.TrainTestKNN()
    ttknn2.predict(pitchContourClassificationModelName,featureVecPredictFolderPath,
                   targetPredictFolderPath,recordingNamesPredict)


    ########################################### representation #########################################################
    for rm in recordingNamesPredict:
        #  filename declaration
        originalPitchtrackFilename = pitchtrackNotePredictFolderPath+rm+'_pitchtrack.csv'
        targetFilename = targetPredictFolderPath+rm+'.json'
        refinedSegmentFeaturesFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSegmentFeatures.json'
        representationFilename = pitchtrackNotePredictFolderPath+rm+'_representation.json'
        figureFilename = pitchtrackNotePredictFolderPath+rm+'_reprensentationContourFigure.png'

        # important txt files!!
        regressionPitchtrackFilename = pitchtrackNotePredictFolderPath+rm+'_regression_pitchtrack.csv'
        refinedSegmentationGroundtruthFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSeg.csv'

        rsm1 = rsm.RefinedSegmentsManipulation()
        rsm1.process(refinedSegmentFeaturesFilename,targetFilename,
                     representationFilename,figureFilename,regressionPitchtrackFilename,
                     originalPitchtrackFilename = originalPitchtrackFilename,
                     refinedSegGroundtruthFilename=refinedSegmentationGroundtruthFilename,
                     slopeTh=slopeTh, flatnoteTh=flatnoteTh)

    runningTime = time.time() - start_time
    # print("--- %s seconds ---" % (time.time() - start_time))

    return runningTime





