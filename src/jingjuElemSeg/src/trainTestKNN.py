################################################## train test ######################################################
import noteClass as nc
import random
import numpy as np
import operator
import json
from sklearn.externals import joblib
from sklearn import neighbors
# import matplotlib.pyplot as plt

class TrainTestKNN(object):

    def __init__(self):
        #  classes = [0,1,2,3,4,5]  #  detail classes
        nc1 = nc.noteClass()
        self.classes = nc1.segmentClasses.values()
        self.featureNamesDict = nc1.featureNamesDict

        self.resetFeatureDictAll()

    def resetFeatureDictAll(self):
        self.featureDictAll = {}
        self.targetDictAll = {}

    def gatherFeatureTarget(self,featureVecFolderPath,targetFolderPath,recordingNames):

        # collect the feature and target from individual file to one file

        self.resetFeatureDictAll()

        for rn in recordingNames:
            featureFilename = featureVecFolderPath+rn+'.json'
            targetFilename = targetFolderPath+rn+'.json'

            with open(featureFilename) as data_file:
                featureDict = json.load(data_file)
            with open(targetFilename) as data_file:
                targetDict = json.load(data_file)

            featureKeyList = list(featureDict.keys())
            targetKeyList = list(targetDict.keys())

            for jj in targetKeyList:
                if jj in featureDict:   #  if jj is also in featureDict keys
                    if None not in featureDict[jj]:
                        self.featureDictAll[rn+'_'+jj] = featureDict[jj]
                        self.targetDictAll[rn+'_'+jj] = targetDict[jj]

    def crossValidation(self,pitchContourClassificationModelName):

        shuffledKeys = list(self.targetDictAll.keys())  #  get the keys which are the recording identifiers
        random.shuffle(shuffledKeys)  #  shuffle the keys

        # partition
        step = len(shuffledKeys)/5

        knn = neighbors.KNeighborsClassifier(n_neighbors=13)

        misclassifiedRateVec = []

        for ii in range(5):
            if ii == 0:
                testKeys = shuffledKeys[ii:step]
            if ii == 4:
                testKeys = shuffledKeys[ii*step:]
            else:
                testKeys = shuffledKeys[ii*step:(ii+1)*step]

            trainKeys = shuffledKeys[:]
            for key2rm in testKeys:
                if key2rm in trainKeys:
                    trainKeys.remove(key2rm)

            trainFeatureVec = []
            testFeatureVec =[]
            trainTargetVec =[]
            testTargetVec = []

            for trk in trainKeys:
                trainFeatureVec.append(self.featureDictAll[trk])
                trainTargetVec.append(self.targetDictAll[trk])
            for tek in testKeys:
                testFeatureVec.append(self.featureDictAll[tek])
                testTargetVec.append(self.targetDictAll[tek])

            # train and test
            knn.fit(trainFeatureVec, trainTargetVec)

            # dump the model
            joblib.dump(knn, pitchContourClassificationModelName)

            testPredictVec = knn.predict(testFeatureVec)

            # result is a vector which non zero elements are misclassfied
            result = np.array(testTargetVec) - np.array(testPredictVec)
            # print result

            # result dictionary contains misclassied key:[target class, classified class]
            resultDict = {}
            for rs in range(len(result)):
                if result[rs]:
                    resultDict[testKeys[rs]] = [testTargetVec[rs],testPredictVec[rs]]
            # print resultDict

            # statistics or misclassfied
            resultStatDict = {}
            for tg in self.classes:
                classesCopy = self.classes[:]
                classesCopy.remove(tg)
                for cl in classesCopy:
                    mcc = str(tg)+str(cl)  #  misclassified combination
                    resultStatDict[mcc] = 0
                    for rd in resultDict.values():
                        if rd == [tg, cl]:
                            resultStatDict[mcc] += 1

            for rsd in resultStatDict.keys():
                resultStatDict[rsd] /= float(len(resultDict))
            # print resultStatDict

            #  sort this dictionary by value
            sortedDict = sorted(resultStatDict.items(), key=operator.itemgetter(1), reverse=True)
            # print sortedDict

            misclassified1to0 = 0
            for jj in resultDict.values():
                if jj[0] == 1 and jj[1] == 0:
                    misclassified1to0 += 1

            misclassifiedRate = len(np.nonzero(result)[0])/float(len(result))
            # misclassifiedRate = misclassified1to0/float(len(np.nonzero(result)[0]))
            misclassifiedRateVec.append(misclassifiedRate)
            print misclassifiedRate
            #  print resultStatDict

        print 'mean misclassified rate: '+ str(np.mean(misclassifiedRateVec))

    def predict(self,pitchContourClassificationModelName, featureVecPredictFolderPath,
                targetPredictFolderPath, recordingNamesPredict):

        '''
        :param pitchContourClassificationModelName: predict knn trained model
        :param featureVecPredictFolderPath: feature vectors to be predicted
        :param targetPredictFolderPath: target path to be output in
        :param recordingNamesPredict: recording names to be predict
        :return: target will be written into json file
        '''

        # load model file
        knn = joblib.load(pitchContourClassificationModelName)

        for rm in recordingNamesPredict:
            # only one recording in list
            featureVecJsonFile = featureVecPredictFolderPath+rm+'.json'
            targetJsonFile = targetPredictFolderPath+rm+'.json'

            # load json feature vectors
            with open(featureVecJsonFile) as data_file:
                data = json.load(data_file)

            keys = data.keys()
            values = data.values()

            # for debug
            # for ii in values:
            #     print ii
            #     print type(ii[0]),type(ii[1]),type(ii[2]),type(ii[3]),type(ii[4]),type(ii[5]),type(ii[6]),type(ii[7]),type(ii[8])

            # classify dictionary
            classification = knn.predict(values)
            classificationDict = {}
            for ii in range(len(keys)):
                classificationDict[keys[ii]] = classification[ii]

            # dump the result as a json
            with open(targetJsonFile, 'w') as outfile:
                json.dump(classificationDict, outfile)


    def featureVec2DPlot(self, dim):

        # plot 2d feature Vectors

        if len(dim) != 2:
            return
        if dim[0] == dim[1]:
            return

        fdaValues = self.featureDictAll.values()
        tdaValues = self.targetDictAll.values()
        fdaValues = np.array(fdaValues)
        tdaValues = np.array(tdaValues)

        colors = ['r','g','b']
        makers = ['o','^','x']
        legends = ['other','vibrato','linear']
        plots = []
        for cl in self.classes:
            index = [i for i, j in enumerate(tdaValues) if j == cl]
            color = colors[cl]*len(index)
            x = fdaValues[index][:,dim[0]]
            y = fdaValues[index][:,dim[1]]
            if dim[0] == 3:
                x = np.log(x)
            if dim[1] == 3:
                y = np.log(y)
            plots.append(plt.scatter(x,y,c=color,marker=makers[cl]))
        plt.legend(plots,legends)

        plt.xlabel(self.featureNamesDict[str(dim[0])])
        plt.ylabel(self.featureNamesDict[str(dim[1])])

        plt.show()
