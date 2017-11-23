# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
import os
from utilFunc import pitch2midi
from utilFunc import cents2pitch
from utilFunc import hz2cents

class pitchtrackSegByNotes(object):

    def __init__(self, fs = 44100, frameSize = 2048, hopSize = 256):
        self.fs = fs
        self.frameSize = frameSize
        self.hopSize = hopSize

        self.max = 0
        self.min = 0

        self.reset()

    def reset(self):
        self.noteStartEndFrame = []
        self.coarseSegmentationStartEndFrame = []
        self.refinedSegmentationStartEndFrame = []
        self.pitchtrackByNotes = []

    def noteEndFrameHelper(self, notePitchtrack, startDur):

        notePitchtrack = np.abs(notePitchtrack)
        notePitchtrack = pitch2midi(notePitchtrack)  # convert to midi note

        self.pitchtrackByNotes.append([notePitchtrack,startDur])
        notePitchtrack = []
        startDur = [0, 0]

        return notePitchtrack, startDur

    def minMaxPitchtrack(self, pitchtrack):

        ptPositive = [item for item in pitchtrack if item > 0]
        self.max = max(ptPositive)
        self.min = min(ptPositive)

        return

    def doSegmentation(self, pitchtrack, monoNoteOut):

        '''
        [ [pitchtrack of note 1, [startingFrame1, durationFrame1]],
                   [pitchtrack of note 2, [startingFrame2, durationFrame2]], ... ... ]

        :param pitchtrack: smoothed pitchtrack output from pYin
        :param monoNoteOut: note pitchtrack output from pYin note transcription
        :return: self.pitchtrackByNotes,
        '''

        # get the max and min values of pitch track
        self.minMaxPitchtrack(pitchtrack)

        # initialisation
        jj = 0
        old_ns = 4
        mnoLen = len(monoNoteOut)
        notePitchtrack = []
        startDur = [0, 0]

        for ii in monoNoteOut:
            ns = ii.noteState
            if (jj == 0 or old_ns == 3) and ns == 1:
                #  note attack on first frame or note attack frame
                notePitchtrack.append(pitchtrack[jj][0])
                startDur[0] = ii.frameNumber
                startDur[1] += 1
            if old_ns == 2 and ns == 3:
                #  note end frame
                notePitchtrack, startDur = \
                    self.noteEndFrameHelper(notePitchtrack, startDur)
            if old_ns == 2 and jj == mnoLen-1:
                #  last frame note and track frame
                notePitchtrack, startDur = \
                    self.noteEndFrameHelper(notePitchtrack, startDur)
            if old_ns == 2 and (ns == 2 or ns == 1):
                #  note stable frame
                notePitchtrack.append(pitchtrack[jj][0])
                startDur[1] += 1

            old_ns = ns
            jj += 1

        return

    def readPyinPitchtrack(self, pitchtrack_filename):

        '''
        :param pitchtrack_filename:
        :return: frameStartingTime, pitchtrack
        '''
        pitchtrack = np.loadtxt(pitchtrack_filename,delimiter=',',usecols=[1,2])
        frameStartingTime = pitchtrack[:,0]
        pitchtrack = pitchtrack[:,1]

        return frameStartingTime, pitchtrack

    def readPyinMonoNoteOut(self, monoNoteOut_filename):

        '''
        :param monoNoteOut_filename:
        :return: noteStartingTime, noteDurationTime
        '''

        monoNoteOut = np.loadtxt(monoNoteOut_filename,delimiter=',',usecols=[1,2,3])
        noteStartingTime = monoNoteOut[:,0]
        noteDurTime = monoNoteOut[:,1]

        return noteStartingTime, noteDurTime

    def pitch2midiPyinMonoNoteOut(self, monoNoteOut_filename,monoNoteOutMidi_filename):

        '''
        read note pitch and convert to midi note
        :param monoNoteOut_filename:
        :return: noteStartingTime, noteDurationTime
        '''

        monoNoteOut = np.loadtxt(monoNoteOut_filename,delimiter=',',usecols=[1,2,3])
        noteStartingTime = monoNoteOut[:,0]
        noteDurTime = monoNoteOut[:,1]
        notePitch = monoNoteOut[:,2]
        notePitchMidi = pitch2midi(notePitch)

        with open(monoNoteOutMidi_filename, 'w+') as outfile:
            outfile.write('startTime'+','
                            +'pitch'+','
                            +'freq'+','
                            +'duration'+','
                            +'noteStr'+'\n')
            for ii in range(len(noteStartingTime)):
                noteCents = hz2cents(float(notePitch[ii]))
                noteStr = cents2pitch(noteCents)
                outfile.write(str(noteStartingTime[ii])+','
                            +str(notePitchMidi[ii])+','
                            +str(notePitch[ii])+','
                            +str(noteDurTime[ii])+','
                            +noteStr+'\n')

        return noteStartingTime, noteDurTime, notePitch, notePitchMidi

    def readCoarseSegmentation(self, coarseSegmentation_filename):

        coarseSeg = np.loadtxt(coarseSegmentation_filename,usecols=[0])

        return coarseSeg

    def doSegmentationForPyinVamp(self, pitchtrack_filename, monoNoteOut_filename):

        # doSegmentationFunction for pYin vamp plugin exported
        # pitchtrack and monoNote

        self.reset()

        frameStartingTime, pitchtrack = self.readPyinPitchtrack(pitchtrack_filename)
        noteStartingTime, noteDurTime = self.readPyinMonoNoteOut(monoNoteOut_filename)

        # convert pitch to midi and save pitch track
        monoNoteOutMidi_filename = monoNoteOut_filename[:-4]+'_midi.csv'
        self.pitch2midiPyinMonoNoteOut(monoNoteOut_filename,monoNoteOutMidi_filename)

        self.minMaxPitchtrack(pitchtrack)

        pitchtrack = np.abs(pitchtrack)
        pitchtrack = pitch2midi(pitchtrack)

        noteEndingTime = noteStartingTime+noteDurTime

        noteStartingIndex = []
        noteEndingIndex = []

        for ii in noteStartingTime:
            noteStartingIndex.append(np.argmin(np.abs(frameStartingTime - ii)))

        for ii in noteEndingTime:
            noteEndingIndex.append(np.argmin(np.abs(frameStartingTime - ii)))

        for ii in range(len(noteStartingIndex)):
            notePitchtrack = pitchtrack[noteStartingIndex[ii]:(noteEndingIndex[ii]+1)]
            startDur = [noteStartingIndex[ii],noteEndingIndex[ii]-noteStartingIndex[ii]+1]

            noteStartingFrame = int(noteStartingTime[ii]*(self.fs/self.hopSize))
            noteEndingFrame = int(noteEndingTime[ii]*(self.fs/self.hopSize))
            self.noteStartEndFrame.append([noteStartingFrame,noteEndingFrame])
            self.pitchtrackByNotes.append([notePitchtrack.tolist(), startDur])

        return

    def coarseSegmentation(self,monoNoteOut_filename, coarseSegmentation_filename):
        '''
        :param monoNoteOut_filename:
        :param coarseSegmentation_filename: segmentation point
        :return:
        '''

        noteStartingTime, noteDurTime = self.readPyinMonoNoteOut(monoNoteOut_filename)
        coarseSegTime = self.readCoarseSegmentation(coarseSegmentation_filename)
        noteEndingTime = noteStartingTime+noteDurTime

        for ii in range(len(noteStartingTime)):
            startingTime_old = noteStartingTime[ii]
            for jj in range(len(coarseSegTime)-1):
                if coarseSegTime[jj]>startingTime_old and coarseSegTime[jj]<noteEndingTime[ii]:

                    if coarseSegTime[jj+1] < noteEndingTime[ii]:

                        segStartingFrame = int(startingTime_old*(self.fs/self.hopSize))
                        segEndingFrame = int(coarseSegTime[jj]*(self.fs/self.hopSize))
                        self.coarseSegmentationStartEndFrame.append([segStartingFrame,segEndingFrame])
                        startingTime_old = coarseSegTime[jj]
                    else:

                        segStartingFrame = int(coarseSegTime[jj]*(self.fs/self.hopSize))
                        segEndingFrame = int(noteEndingTime[ii]*(self.fs/self.hopSize))
                        self.coarseSegmentationStartEndFrame.append([segStartingFrame,segEndingFrame])

        return

    def refinedSegmentation(self,refinedSegmentation_filename):
        '''
        :param refinedSegmentation_filename:
        :return: refined segmentation start end frame
        '''

        noteStartingTime, noteDurTime = self.readPyinMonoNoteOut(refinedSegmentation_filename)
        noteEndingTime = noteStartingTime+noteDurTime

        for ii in range(len(noteStartingTime)):
            noteStartingFrame = int(noteStartingTime[ii]*(self.fs/self.hopSize))
            noteEndingFrame = int(noteEndingTime[ii]*(self.fs/self.hopSize))
            self.refinedSegmentationStartEndFrame.append([noteStartingFrame,noteEndingFrame])

    def pltNotePitchtrack(self, saveFig = False, figFolder = './'):

        '''
        :param notePitchtrack: [pitch1, pitch2, ...]
        :param startDur: [starting frame, duration frame]
        :return:
        '''

        if not os.path.exists(figFolder):
            os.makedirs(figFolder)

        jj = 1
        for ii in self.pitchtrackByNotes:
            notePitchtrack = ii[0]
            startDur = ii[1]

            # time in s
            startingTime = (self.frameSize/2 + self.hopSize*startDur[0])/float(self.fs)
            durTime = (startDur[1]+1)*self.hopSize/float(self.fs)
            frameRange = range(startDur[0], startDur[0]+startDur[1])

            plt.figure()
            plt.plot(frameRange, np.abs(notePitchtrack))
            plt.ylabel('midi note number, 69: A4')
            plt.xlabel('frame')
            plt.title('starting time: ' + str(startingTime) +
                      ' duration: ' + str(durTime))

            axes = plt.gca()
            #axes.set_xlim([xmin,xmax])
            #axes.set_ylim([self.min-5,self.max+5])

            if saveFig == True:
                plt.savefig(figFolder+str(jj)+'.png')
                plt.close()
            jj += 1

        if saveFig == False:
            plt.show()