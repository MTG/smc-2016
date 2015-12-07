# -*- coding: utf-8 -*-

import sys, os
import matplotlib.pyplot as plt
import time
import json
import shutil
import random

# add pYin src path
pYinPath = '../pypYIN/src'
sys.path.append(pYinPath)

# add src path
dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)

import pitchtrackSegByNotes
import essentia.standard as ess
import numpy as np
import noteClass as nc
import featureVecTarget as fvt
import trainTestKNN as ttknn
import refinedSegmentsManipulation as rsm
import evaluation as evalu
from vibrato import vibrato
from pYINPtNote import pYINPtNote

if __name__ == "__main__":

    start_time = time.time()  # starting time

    '''
    ######################################### pYIN pitchtrack and notes ################################################
    filename_amateur = '/Users/gong/Documents/MTG document/Jingju arias/jingjuElementMaterials/laosheng/test/weiguojia_section_amateur.wav'
    filename_pro = '/Users/gong/Documents/MTG document/Jingju arias/jingjuElementMaterials/laosheng/test/weiguojia_section_pro.wav'
    pYINPtNote(filename_amateur)
    #filename1 = pYinPath + '/testAudioLong.wav'
    '''

    ############################################## initialsation #######################################################
    pitchtrackNoteTrainFolderPath = './pYinOut/laosheng/train/'

    # classification training ground truth
    groundtruthNoteLevelPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/pyinNoteCurvefit/classified'
    groundtruthNoteDetailPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/refinedSegmentCurvefit/classified'

    # classification model path
    pitchContourClassificationModelName = './pYinOut/laosheng/train/model/pitchContourClassificationModel.pkl'

    # feature, target folders
    featureVecTrainFolderPath = './pYinOut/laosheng/train/featureVec/'
    targetTrainFolderPath = './pYinOut/laosheng/train/target/'

    # predict folders
    pitchtrackNotePredictFolderPath = './pYinOut/laosheng/predict/'
    featureVecPredictFolderPath = './pYinOut/laosheng/predict/featureVec/'
    targetPredictFolderPath = './pYinOut/laosheng/predict/target/'


    recordingNamesTrain = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    recordingNamesPredict = ['weiguojia_section_pro','weiguojia_section_amateur']

    evaluation = False                      #  parameters grid search

    slopeTh = 60.0                          #  contour combination slope difference threshold
    flatnoteTh = 80.0                       #  threshold for deciding one note as flat pitch note
    #recordingNamesPredict = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']       # evaluation

    '''
    ############################################## train process #######################################################
    # WARNING!!! don't run train process ANY MORE!!! because the segmentRefinement function is not the same ANY MORE!
    # If this is ran, we need MANUALLY re-prepare the training groundtruch!!! This will take several days!!!
    ######## segmentation and features ########
    nc1 = nc.noteClass()
    nc1.noteSegmentationFeatureExtraction(pitchtrackNoteTrainFolderPath,featureVecTrainFolderPath,recordingNamesTrain,segCoef=0.2)

    ######### construct target json ###########

    fvt1 = fvt.FeatureVecTarget()
    fvt1.constructJson4DetailFeature(featureVecTrainFolderPath,targetTrainFolderPath,recordingNamesTrain,groundtruthNoteDetailPath)

    ################ train ####################
    ttknn1 = ttknn.TrainTestKNN()
    ttknn1.gatherFeatureTarget(featureVecTrainFolderPath,targetTrainFolderPath,recordingNamesTrain)
    # ttknn1.featureVec2DPlot([3,7])
    ttknn1.crossValidation(pitchContourClassificationModelName)
    '''

    ################################################## predict #########################################################
    # segmentation
    nc2 = nc.noteClass()
    nc2.noteSegmentationFeatureExtraction(pitchtrackNotePredictFolderPath,
                                          featureVecPredictFolderPath,
                                          recordingNamesPredict,
                                          segCoef=0.3137,predict=True)

    # predict
    ttknn2 = ttknn.TrainTestKNN()
    ttknn2.predict(pitchContourClassificationModelName,featureVecPredictFolderPath,
                   targetPredictFolderPath,recordingNamesPredict)

    '''
    ################################################ evaluation code ###################################################
    # uncomment it only when needs evaluation
    with open('./pYinOut/laosheng/predict/evaluationResult02.txt', "w") as outfile:
        for sc in np.linspace(0.2,0.5,30):
            COnOffF,COnF,OBOnRateGT,OBOffRateGT = nc2.noteSegmentationFeatureExtraction(pitchtrackNotePredictFolderPath,
                                                  featureVecPredictFolderPath,recordingNamesPredict,
                                                  segCoef=0.3137,predict=True,evaluation=True)
            outfile.write(str(sc)+'\t'+str([COnOffF,COnF,OBOnRateGT,OBOffRateGT])+'\n')
    '''

    ########################################### representation #########################################################
    for rm in recordingNamesPredict:
        #  filename declaration
        targetFilename = targetPredictFolderPath+rm+'.json'
        refinedSegmentFeaturesFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSegmentFeatures.json'
        representationFilename = pitchtrackNotePredictFolderPath+rm+'_representation.json'
        figureFilename = pitchtrackNotePredictFolderPath+rm+'_reprensentationContourFigure.png'

        # important txt files!!
        pitchtrackFilename = pitchtrackNotePredictFolderPath+rm+'_regression_pitchtrack.txt'
        refinedSegmentationGroundtruthFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSeg.txt'

        rsm1 = rsm.RefinedSegmentsManipulation()
        rsm1.process(refinedSegmentFeaturesFilename,targetFilename,
                     representationFilename,figureFilename,pitchtrackFilename,
                     refinedSegGroundtruthFilename=refinedSegmentationGroundtruthFilename,
                     slopeTh=slopeTh, flatnoteTh=flatnoteTh)

    '''
    ############################################### evaluation #########################################################
    evalu1 = evalu.Evaluation()
    with open('./pYinOut/laosheng/predict/evaluationResultRefined.txt', "w") as outfile:
        #  grid search
        for slopeTh in range(0,110,10):
            for flatnoteTh in range(0,110,10):

                if evaluation:
                    COnOffall, COnall, OBOnall, OBOffall,gt,st = 0,0,0,0,0,0        # evaluation metrics

                for rm in recordingNamesPredict:
                    #  filename declaration
                    targetFilename = targetPredictFolderPath+rm+'.json'
                    refinedSegmentFeaturesFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSegmentFeatures.json'
                    # representationFilename = pitchtrackNotePredictFolderPath+rm+'_representation.txt'
                    representationFilename = pitchtrackNotePredictFolderPath+rm+'_representation.json'
                    figureFilename = pitchtrackNotePredictFolderPath+rm+'_reprensentationContourFigure.png'
                    pitchtrackFilename = pitchtrackNotePredictFolderPath+rm+'_regression_pitchtrack.txt'

                    if evaluation:
                        refinedSegmentationGroundtruthFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSeg.txt'
                    else:
                        refinedSegmentationGroundtruthFilename = None

                    rsm1 = rsm.RefinedSegmentsManipulation()
                    rsm1.process(refinedSegmentFeaturesFilename,targetFilename,
                                 representationFilename,figureFilename,pitchtrackFilename,
                                 refinedSegGroundtruthFilename=refinedSegmentationGroundtruthFilename,
                                 slopeTh=slopeTh, flatnoteTh=flatnoteTh)

                    #  evaluation metrics collection
                    if evaluation:                                                  #  if refined seg file exist
                        COnOffall += rsm1.evaluationMetrics[0]
                        COnall += rsm1.evaluationMetrics[1]
                        OBOnall += rsm1.evaluationMetrics[2]
                        OBOffall += rsm1.evaluationMetrics[3]
                        gt += rsm1.evaluationMetrics[4]
                        st += rsm1.evaluationMetrics[5]

                if evaluation:
                    COnOffF,COnF,OBOnRateGT,OBOffRateGT = evalu1.metrics(COnOffall,COnall,OBOnall,OBOffall,gt,st)
                    outfile.write(str(slopeTh)+'\t'+str(flatnoteTh)+'\t'+
                                  str(COnOffF)+'\t'+str(COnF)+'\t'+
                                  str(OBOnRateGT)+'\t'+str(OBOffRateGT)+'\n')
                    print slopeTh,flatnoteTh,COnOffF,COnF,OBOnRateGT,OBOffRateGT
    '''

    '''
    ################################################## copy code #######################################################
    #  below code to 1) get class note name from classified folder, 2) copy the midinote png with the same name into
    #  classified + 'midinote' folder
    nc1 = ncc.noteClass()

    prependPathClassified = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/classified'
    prependPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train'

    recordingNamesClassified = ['male_02_neg_01', 'male_12_neg_01', 'male_12_pos_01', 'male_13_pos_01']
    recordingNames = ['male_02/neg_1_midinote', 'male_12/neg_1_midinote', 'male_12/pos_1_midinote', 'male_13/pos_1_midinote']

    for ii in range(len(recordingNames)):
        ncd = nc1.noteNumberOfClass(prependPathClassified, recordingNamesClassified[ii])

        pathnameMO = os.path.join(prependPath, recordingNames[ii])
        onlypngs = [ f for f in os.listdir(pathnameMO) if f.endswith('.png') ]

        for key in ncd:

            #  path to move into
            pathnameMI = os.path.join(prependPathClassified, key, recordingNamesClassified[ii]+'midinote')

            if not os.path.exists(pathnameMI):
                os.makedirs(pathnameMI)

            for png in onlypngs:
                print png
                if png in ncd[key]:
                    shutil.copyfile(os.path.join(pathnameMO, png), os.path.join(pathnameMI, png))
    '''


    print("--- %s seconds ---" % (time.time() - start_time))





