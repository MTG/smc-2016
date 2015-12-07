########################################## feature vectors classify ################################################

## this class constructs featureVec and target json file
import noteClass as nc
import os,json

class FeatureVecTarget(object):

    def constructJson4NoteFeature(self,featureVecFolderPath,targetFolderPath,recordingNames,groundtruthNoteLevel):

        #  note level classification groundtruth json

        nc1 = nc.noteClass()
        allNoteClasses = list(nc1.basicNoteClasses.keys()) + nc1.unClassifiedNoteClasses

        for rn in recordingNames:
            featureFilename = featureVecFolderPath+rn+'.json'
            targetFilename = targetFolderPath+rn+'.json'
            with open(featureFilename) as data_file:
                featureDict = json.load(data_file)

            targetDict = {}
            for ii in range(1, len(featureDict)+1):
                for noteClass in allNoteClasses:
                    noteClassRecordingFoldername = os.path.join(groundtruthNoteLevel, noteClass, rn+'midinote')
                    if os.path.isdir(noteClassRecordingFoldername):
                        onlypngs = [ f for f in os.listdir(noteClassRecordingFoldername) if f.endswith('.png') ]
                        for png in onlypngs:
                            pngNum = os.path.splitext(png)[0]
                            if str(ii) == pngNum:
                                if noteClass in nc1.basicNoteClasses:
                                    #  targetDict[ii] = nc1.basicNoteClasses[noteClass]  #  detail classes
                                    targetDict[ii] = 0  # 5 basic classes
                                else:
                                    #  targetDict[ii] = 5  #  detail classes
                                    targetDict[ii] = 1  # non classified classes

            with open(targetFilename, 'w') as outfile:
                json.dump(targetDict, outfile)

    def constructJson4DetailFeature(self, featureVecFolderPath,targetFolderPath,recordingNames,groundtruthCSVPath):

        #  detail level classification groundtruth json

        nc1 = nc.noteClass()
        segmentClasses = nc1.segmentClasses

        for rn in recordingNames:
            featureFilename = featureVecFolderPath+rn+'.json'
            targetFilename = targetFolderPath+rn+'.json'

            CSVFile = ''.join([str(groundtruthCSVPath),'/',rn,'.csv'])

            with open(featureFilename) as data_file:
                featureDict = json.load(data_file)

            # read classification groundtruth csv file
            with open(CSVFile) as groundtruthFile:
                targetDict = {}
                for line in groundtruthFile.readlines():
                    splitLine = line.split(',')
                    for iic in range(len(splitLine)):
                        c = splitLine[iic].rstrip('\n')
                        if c != '':  #  empty string is False
                            cInt = int(c)
                            targetDict[cInt] = iic

            with open(targetFilename, 'w') as outfile:
                json.dump(targetDict, outfile)
