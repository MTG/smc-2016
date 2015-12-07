import numpy as np
import json
import matplotlib.pyplot as plt

class ConcatenateSegment(object):

    def __init__(self):
        self.fs = 44100.0
        self.hopSize = 256.0

    ########################################## notes manipulation ######################################################

    def readPyinMonoNoteOut(self, monoNoteOut_filename):

        '''
        :param monoNoteOut_filename:
        :return: noteStartingTime, noteDurationTime
        '''

        monoNoteOut = np.loadtxt(monoNoteOut_filename,delimiter=',',usecols=[1,2,3])
        noteStartingTime = monoNoteOut[:,0]
        noteDurTime = monoNoteOut[:,1]
        pitch = monoNoteOut[:,2]
        midiNote = self.pitch2midi(pitch)

        return noteStartingTime, noteDurTime, midiNote

    def getNoteFrameBoundary(self,noteStartingTime, noteDurTime):
        '''
        :param
        :return: start end frame
        '''

        noteEndingTime = noteStartingTime+noteDurTime

        noteStartingFrame = np.round(noteStartingTime*(self.fs/self.hopSize)).astype(int)
        noteEndingFrame = np.round(noteEndingTime*(self.fs/self.hopSize)).astype(int)

        return noteStartingFrame, noteEndingFrame

    def generateNotePitchtrack(self,noteStartingFrame, noteEndingFrame, midiNote):
        '''
        :param noteStartingFrame:
        :param noteEndingFrame:
        :param midiNote:
        :return: notePts: concatenated note pitchtrack,
        noteMap: mapping index from concatenateFrame to original Frame
        '''

        notePts = []
        noteFrameMap = []
        concatenateFrame = 0

        noteStartingFrameTemp = 0
        noteStartingFrameConcatenate = []
        noteEndingFrameConcatenate = []

        for ii in range(len(noteStartingFrame)):
            noteFrame = range(noteStartingFrame[ii],noteEndingFrame[ii]+1)
            noteMidi = np.ones(len(noteFrame))*midiNote[ii]
            noteMidi = noteMidi.tolist()
            notePts = notePts+noteMidi

            noteEndingFrameTemp = noteStartingFrameTemp+len(noteFrame)-1
            noteStartingFrameConcatenate.append(noteStartingFrameTemp)
            noteEndingFrameConcatenate.append(noteEndingFrameTemp)

            noteStartingFrameTemp = noteStartingFrameTemp+len(noteFrame)

            # for jj in range(len(noteFrame)):
            #     startFrame = noteStartingFrame[ii]
            #     currentFrame = startFrame+jj
            #     noteFrameMap.append([concatenateFrame,currentFrame])
            #     concatenateFrame += 1

        return notePts, noteStartingFrameConcatenate, noteEndingFrameConcatenate

    ########################################## segments manipulation ###################################################

    def readRepresentation(self,filename):

        with open(filename) as data_file:
            refinedSegmentFeatures = json.load(data_file)

        # separate segment features
        segmentPts = refinedSegmentFeatures['pitchcontours']
        boundaries = refinedSegmentFeatures['boundary']
        target = refinedSegmentFeatures['target']
        features = refinedSegmentFeatures['Features']

        return segmentPts,boundaries,target

    def concatenate(self,segmentPts):

        '''
        :param segmentPts:
        :param boundaries:
        :return: concatenate pitchtrack; mapping from concatenation index to original index
        '''

        keys = sorted([int(x) for x in segmentPts.keys()])

        concatenatePts = []
        concatenateMap = []
        concatenateFrame = 0

        segStartingFrameTemp = 0
        segStartingFrameConcatenate = []
        segEndingFrameConcatenate = []

        for k in keys:
            ks = str(k)
            concatenatePts = concatenatePts+segmentPts[ks]

            segEndingFrameTemp = segStartingFrameTemp+len(segmentPts[ks])-1
            segStartingFrameConcatenate.append(segStartingFrameTemp)
            segEndingFrameConcatenate.append(segEndingFrameTemp)

            segStartingFrameTemp = segStartingFrameTemp+len(segmentPts[ks])

            # for jj in range(len(segmentPts[ks])):
            #     startFrame = boundaries[ks][0]
            #     currentFrame = startFrame+jj
            #     concatenateMap.append([concatenateFrame,currentFrame])
            #     concatenateFrame += 1

        return concatenatePts,segStartingFrameConcatenate,segEndingFrameConcatenate

    ################################################# public methods ###################################################

    def pitchtrackNormalization(self, pitchtrack):

        '''
        normalize pitch to range [0,1]
        :param pitchtrack:
        :return:
        '''
        notePtNorm = np.array(pitchtrack[:])
        notePtNorm = notePtNorm-min(notePtNorm)
        notePtNorm = notePtNorm/max(notePtNorm)

        return notePtNorm.tolist()

    def resampling(self, notePts, noteStartingFrame, noteEndingFrame, lenNotePts_target):
        '''
        :param notePts_s:
        :param lenNotePts_t:
        :return:
        '''

        # interpolation to get more points in shorter pitchtrack
        lenNotePts = len(notePts)
        x = np.linspace(0,lenNotePts-1,lenNotePts)
        xvals = np.linspace(0,lenNotePts-1,lenNotePts_target)
        notePts_interp = np.interp(xvals, x, notePts)

        # get the note boundray frames
        noteStartingFrameInterp = []
        noteEndingFrameInterp = []
        for ii in range(len(noteStartingFrame)):
            noteStartingFrameInterp.append(np.argmin(np.abs(noteStartingFrame[ii]-xvals)))
            noteEndingFrameInterp.append(np.argmin(np.abs(noteEndingFrame[ii]-xvals)))


        return notePts_interp, noteStartingFrameInterp, noteEndingFrameInterp

    def pitch2midi(self, pitchtrack):
    #  convert pitch hz to midi note number

        flag = False
        if not isinstance(pitchtrack, np.ndarray):
            pitchtrack = np.array(pitchtrack)
            flag = True

        midi = 12.0 * np.log(pitchtrack/440.0)/np.log(2.0) + 69.0

        # if flag:
        #     midi = midi.tolist()

        return midi