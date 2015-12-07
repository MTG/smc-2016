import os,json
import numpy as np
import matplotlib.pyplot as plt
import utilFunc as uf
import evaluation as evalu
import pitchtrackSegByNotes
import vibrato
from scipy.signal import argrelextrema
from sklearn import linear_model

#import statsmodels.api as smapi


class noteClass(object):

    def __init__(self):
        self.basicNoteClasses = {'glissando':0, 'cubic':1,
                            'parabolic':2, 'vibrato':3, 'other':4}

        #  unClassifiedClass: 5
        self.unClassifiedNoteClasses = ['flat+semiVibrato', 'semiVibrato+flat',
                                        'semiVibrato+flat+semiVibrato', 'nonClassified']

        self.segmentClasses = {'linear':0, 'vibrato':2, 'other':1}

        self.featureNamesDict = {'0':'slope','1':'intercept','2':'rsquared','3':'polyVar','4':'standard deviation',
                                 '5':'extrema amount','6':'contour length','7':'fitting curve zero crossing','8':'vibrato frequency'}
        self.samplerate = 44100.0
        self.hopsize = 256.0
        self.resetLocalMinMax()
        self.resetDiffExtrema()
        self.resetSegments()
        self.resetRefinedNotePts()

    def resetLocalMinMax(self):
        self.minimaInd = []
        self.maximaInd = []
        self.ySmooth = []

    def resetDiffExtrema(self):
        self.diffX = []
        self.diffAmp = []
        self.diffFusion = []
        self.extrema = []

    def resetSegments(self):
        self.segments = []
        self.threshold = 0

    def resetRefinedNotePts(self):
        self.refinedNotePts = []

    def noteNumberOfClass(self, prependPath, recordingName):

        '''
        :param prependPath: is the pathname prepend to note class name
        :param recordingPathName: is the pathname of recording
        :return:
        '''
        noteClassDict = {}
        for noteClass in self.basicNoteClasses:
            noteClassPath = os.path.join(prependPath, noteClass, recordingName)
            onlypngs = [ f for f in os.listdir(noteClassPath) if f.endswith('.png') ]
            noteClassDict[noteClass] = onlypngs

        return noteClassDict

    def normalizeNotePt(self, notePt):
        '''
        :param notePt: note pitch contour
        :return: x, x-axis series [0,1]; y, y-axis series [0,1]
        '''

        x = np.linspace(0,1,len(notePt))
        notePtNorm = notePt[:]
        #notePtNorm = notePtNorm-min(notePtNorm)
        #notePtNorm = notePtNorm*2/max(notePtNorm)

        #  remove DC
        notePtNorm = notePtNorm - np.mean(notePtNorm)

        return x, notePtNorm

    def pitchContourFit(self, x, y, deg):

        '''
        :param x: x support [0,1]
        :param y: y support [0,1]
        :return: polynomial coef p, residuals, rank, singular_values, rcond
        '''

        if len(y) <= deg+1:  # don't fit the curve if it's too short!
            return

        p= np.polyfit(x=x, y=y, deg=deg, full=False)
        return p

    def pitchContourLMBySM(self,x,y):

        # use statsmodel do linear regression, 1-d.
        # this implementation can detect outliers easily

        if len(x)-1-1 <= 0:
            return None,None,None

        # statsmodel dependency not support in server
        '''
        X = smapi.add_constant(x, prepend=False)        #  add intercept

        ''
        mod = smapi.RLM(y, X)
        res = mod.fit()
        r2_wls = smapi.WLS(mod.endog, mod.exog, weights=res.weights).fit().rsquared
        print r2_wls, res.params
        ''

        regression = smapi.OLS(y,X).fit()               #  fit model

        # outliers test method 1
        # test = regression.outlier_test()                              #
        # outliers = [ii for ii,t in enumerate(test) if t[2] < 0.5]     # outliers test

        # outliers test method 2: cook's distance
        influence = regression.get_influence()
        (c, p) = influence.cooks_distance               # cook's distance
        threshold = 4.0/(len(x)-1-1)
        outliers = [ii for ii,t in enumerate(c) if t > threshold]

        outliers.sort()

        # remove outliers
        if len(outliers):
            xr = np.delete(x, outliers)
            yr = np.delete(y, outliers)
            yinterp = np.interp(x, xr, yr)                         #  linear interpolation outliers

            regression = smapi.OLS(yinterp,X).fit()               #  fit model
        else:
            yinterp = y

        return regression.params, regression.rsquared, yinterp
        '''

        # scikit learn robust regresssion
        # reshape for scikit learn data structure
        x_reshape = np.reshape(x, (len(x),1))
        y_reshape = np.reshape(y, (len(y),1))
        # Robustly fit linear model with RANSAC algorithm
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(x_reshape, y_reshape)
        inlier_mask = model_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        slope = model_ransac.estimator_.coef_
        slope = slope[0][0]

        intercept =  model_ransac.estimator_.intercept_
        intercept = intercept[0]

        p = [slope, intercept]

        # outliers
        outliers = [ii for ii,t in enumerate(outlier_mask) if t]
        outliers.sort()

        # remove outliers
        if len(outliers):
            xr = np.delete(x, outliers)
            yr = np.delete(y, outliers)
            yinterp = np.interp(x, xr, yr)                         #  linear interpolation outliers
        else:
            yinterp = y

        # rsquared
        rsquared = self.polyfitRsquare(x,yinterp,p)

        return np.array(p), rsquared, yinterp

        # print regression.params
        # print regression.rsquared
        # print 'Outliers: ', list(outliers)

    def polyfitRsquare(self, x, y, p):

        if p is None:
            return

        # r-squared
        pc = np.poly1d(p)
        # fit values, and mean
        yhat = pc(x)                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        sserr = np.sum((y-yhat)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y-ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        rsqure = 1-sserr/sstot

        return rsqure

    def polyfitVariance(self,x,y,p):

        # the variance of the fitting

        if p is None:
            return

        pc = np.poly1d(p)
        yhat = pc(x)
        var = np.square(np.sum((y-yhat)**2)/len(y))

        return var

    def polyfitResiduals(self,y,deg):

        x = np.linspace(0,1,len(y))
        p = self.pitchContourFit(x, y, deg)         #  use higher degrees curve fitting
        pc = np.poly1d(p)
        residuals = y-pc(x)                         #  residuals of curve fitting
        residuals = residuals-np.mean(residuals)    #  DC remove

        return residuals

    def vibFreq(self, y, deg):

        if len(y) <= deg+1:  #  not calculating the vib freq if it's too short
            return None, None

        # calculate the vib frequency
        # y should not be the pitch track without normalizing

        residuals = self.polyfitResiduals(y,deg)

        freq = uf.vibFreq(residuals, self.samplerate, self.hopsize)

        return freq, residuals

    def vibExt(self, residuals, vibRate):

        if residuals is None:
            return

        ext = uf.vibExt(residuals, self.samplerate, self.hopsize, vibRate)

        return ext

    def vibratoCoef(self,rpt,vibB,vibBExt,vibBFreq):
        '''
        vibrato coefs by vibrato Nadine
        :return:
        '''
        vibLen = 0
        vib = False
        vibMExt = 0
        vibMFreq = 0

        # vibrato frame length
        for vb in vibBExt:
            vibLen += len(vb)

        # if vibrato length is greater than 0.1, it's a vibrato
        if vibLen>len(rpt)*0.2:
            vib = True
            vibExt = []
            vibFreq = []

            for ii in range(len(vibBExt)):
                vibExt = vibExt+ vibBExt[ii]
                vibFreq = vibFreq + vibBFreq[ii]
            vibMExt = np.mean(vibExt,dtype=np.float)
            vibMFreq = np.mean(vibFreq,dtype=np.float)

        vibOut = [vib,vibMFreq,vibMExt]

        return vibOut

    ######################################## minima and maxima treatment ###############################################

    def localMinMax(self, y, box_pts=None):

        #  detect local minima and maxima using the smoothed curve

        self.resetLocalMinMax()

        leny = len(y)
        # smooth the signal
        # print box_pts,len(y)
        if box_pts is None:             # detail segmentation box
            n = int(leny/10)
            box_pts = 3 if n<3 else n
        if leny>52 and box_pts>=leny:   # segment > 0.3s but box is too large
            n = int(leny/5)
            box_pts = n
        if box_pts < leny:
            self.ySmooth = uf.smooth(y, box_pts)
        half_box_pts = np.ceil(box_pts/2.0)

        if len(self.ySmooth):
            # for local maxima
            self.maximaInd = argrelextrema(self.ySmooth, np.greater)
            # remove the boundary effect of convolve
            self.maximaInd = [mi for mi in self.maximaInd[0] if (mi>half_box_pts and mi<leny-half_box_pts)]

            # for local minima
            self.minimaInd = argrelextrema(self.ySmooth, np.less)
            # remove the boundary effect of convolve
            self.minimaInd = [mi for mi in self.minimaInd[0] if (mi>half_box_pts and mi<leny-half_box_pts)]

        return box_pts

    def extremaAmount(self):
        return len(self.minimaInd)+len(self.maximaInd)

    def fittingcurveCrossing(self,x,y,p):

        #  zero crossing fitting curve version

        if p is None:
            return

        pc = np.poly1d(p)
        yhat = pc(x)

        fcc = 0
        for ii in range(len(x)-1):
            if y[ii]-yhat[ii]>0 and y[ii+1]-yhat[ii+1]<=0:  #  from + to -
                fcc += 1
            if y[ii]-yhat[ii]<0 and y[ii+1]-yhat[ii+1]>=0:  #  from - to +
                fcc += 1

        return fcc

    def diffExtrema(self, x, y):

        #  the absolute amplitudes difference of consecutive extremas

        self.resetDiffExtrema()

        if (len(self.minimaInd) == 0 and len(self.maximaInd) == 0) or len(self.ySmooth) == 0:
            print 'No minima and maxima detected!'
            return

        #  extrema indices with beginning and ending indices
        self.extrema = [0] + sorted(self.minimaInd+self.maximaInd) + [len(self.ySmooth)-1]

        for ii in range(1,len(self.extrema)):
            da = abs(self.ySmooth[self.extrema[ii]]-self.ySmooth[self.extrema[ii-1]])
            dx = self.extrema[ii]-self.extrema[ii-1]
            self.diffAmp.append(da)
            self.diffX.append(dx)

        #  normalize diffX to [0,1] interval
        diffX = np.array(self.diffX)
        if len(np.unique(diffX)) != 1:  #  if array only contains one unique element, don't do subtraction
            diffX = diffX-np.min(diffX)
        diffX = diffX/float(np.max(diffX))

        self.diffX = diffX.tolist()

        #  the boundary of difference are treated differently
        self.diffAmp[0] = abs(self.ySmooth[self.extrema[1]]-y[0])
        self.diffAmp[-1] = abs(y[-1]-self.ySmooth[self.extrema[-2]])

        diffFusion = np.array(self.diffAmp)*np.array(self.diffX)
        self.diffFusion = diffFusion.tolist()

    def consecutiveSections(self, indices):

        # find consecutive sections of a list
        # example: [1,2,3,5,6,7], return [[1,2,3],[5,6,7]]

        conSections = []
        lenIn = len(indices)
        if lenIn > 1:
            conSec = [indices[0]]
            for ii in range(1,lenIn):
                if ii == lenIn-1:  #  last index
                    if indices[ii] == indices[ii-1]+1:
                        conSec.append(indices[ii])
                        conSections.append(conSec)
                    else:
                        conSections.append(conSec)
                        conSections.append([indices[ii]])
                elif indices[ii] == indices[ii-1]+1:
                    conSec.append(indices[ii])
                else:
                    conSections.append(conSec)
                    conSec = [indices[ii]]
        elif lenIn == 1:
            conSections.append(indices)

        return conSections

    def segmentPointDetection1(self, threshold):

        #  detection of the segmentation point by extrema cumulation

        # threshold /= len(diffFusion)
        #print threshold

        lenDiffAmp = len(self.diffAmp)
        self.resetSegments()

        if not lenDiffAmp:
            return

        if lenDiffAmp <= 3:  #  we don't consider if there is only one or two extremas case
            return

        cumulDiff = [self.diffAmp[0]]
        for ii in range(1,lenDiffAmp):
            cumulDiff.append(self.diffAmp[ii])
            if np.std(cumulDiff) > threshold:
                self.segments.append(ii)
                cumulDiff = []

    def segmentPointDetection2(self):

        #  detection of the segmentation point by setting a general y-axis threshold

        lenDiffAmp = len(self.diffAmp)
        self.resetSegments()

        if not lenDiffAmp:
            return

        thresholds = []
        if lenDiffAmp > 3:  #  we don't consider if there is only one or two extremas case
            sortedDiffAmp = sorted(self.diffAmp)
            thresholds = sortedDiffAmp[:]

        smallestSdConSecs = [range(0,lenDiffAmp-1)]
        if len(thresholds):
            smallestSd = float("inf")
            for th in thresholds:
                lessThIndices = [n for n,i in enumerate(self.diffAmp) if i<th ]
                largerThIndices = [n for n,i in enumerate(self.diffAmp) if i>=th ]

                # find consecutive sections
                letiConSecs = self.consecutiveSections(lessThIndices[:])
                latiConSecs = self.consecutiveSections(largerThIndices[:])
                conSecs = letiConSecs+latiConSecs

                # check if singleton in section
                singleton = False
                for cs in conSecs:
                    if len(cs) == 1 and cs[0] != 0 and cs[0] != lenDiffAmp-1:
                        singleton = True
                        break

                if not singleton:
                    # mean std
                    sds = []
                    for cs in conSecs:
                        diffAmp = np.array(self.diffAmp)
                        sd = np.std(diffAmp[cs])
                        sds.append(sd)
                    std = np.mean(sds)

                    # smallest std consective sections
                    if std<smallestSd:
                        smallestSd = std
                        smallestSdConSecs = conSecs
                        self.threshold = th

        # segments of extrema index
        for cs in smallestSdConSecs:
            seg = cs[:]
            # seg.append(seg[-1]+1)
            self.segments.append(seg[0])
        self.segments = sorted(self.segments)
        if len(self.segments) > 1:
            self.segments = self.segments[1:]

    def segmentRefinement(self, notePt):

        # construct the refined segmentation pitch contours

        self.resetRefinedNotePts()

        if len(self.segments):                              #  note segmented by extremas
            extrema = np.array(self.extrema)
            segments = extrema[self.segments]
            segments = np.insert(segments, 0, 0)
            segments = np.append(segments, len(notePt))
            for ii in range(1,len(segments)):
                pt = notePt[segments[ii-1]:segments[ii]]
                self.refinedNotePts.append(pt)
        else:                                               #  no extremas
            self.refinedNotePts.append(notePt)


    def pltNotePtFc(self, x, y, p, rsquare, vibFreq, saveFig=False, figFolder='./', figNumber=0):

        '''
        plot the pitchtrack and the fitting curve
        '''

        if not os.path.exists(figFolder):
            os.makedirs(figFolder)

        #plt.figure()  #  create a new figure
        ######  pitchtrack figure
        f, ax = plt.subplots()
        ax.plot(x, y, 'b.', label='note pitchtrack')
        if len(self.ySmooth):
            ax.plot(x, self.ySmooth, 'b--', label='smoothed pitchtrack')

        if p is not None:
            pc = np.poly1d(p) # polynomial class
            ax.plot(x, pc(x), 'r-', label='fitting curve')

        '''
        #  draw vibrato part
        if len(vibB) >= 2:
            # sampleRate = 44100/256
            # step = 13

            for ii in range(len(vibB)/2):
                st = vibB[ii*2]
                end = vibB[ii*2+1]
                ax.plot(x[st:end], y[st:end], 'r.')
                #  ax.annotate(str(vibBFreq[0][ii*step:ii*step+1]), xy = (x[st], 0),
                #        xytext=(x[st], 0.05))
        '''

        if len(self.minimaInd) or len(self.maximaInd):
            ax.plot(x[self.minimaInd], self.ySmooth[self.minimaInd], 'gv', markersize=10.0)  #  markers of local minimas
            ax.plot(x[self.maximaInd], self.ySmooth[self.maximaInd], 'g^', markersize=10.0)  #  markers of local maximas

            #  add text above the extreme
            sortedIndices = sorted(self.minimaInd+self.maximaInd)
            for ii in range(len(sortedIndices)):
                ax.annotate(str(ii), xy = (x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]),
                            xytext=(x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]+0.05))

            if len(self.segments):
                for seg in self.segments:
                    ax.axvline(x[self.extrema[seg]], linestyle='--')

        # axarr[0].legend(loc='best')
        ax.set_ylabel('normed midinote number')
        # plt.xlabel('frame')
        ax.set_title('rsquare: '+str(rsquare)+'vibratoFreq: '+str(vibFreq))

        if saveFig==True:
            plt.savefig(figFolder+str(figNumber)+'.png')
            plt.close(f)  #  close the subplot close(f), plt.close() is to close plt

        #####  difference figures
        if len(self.diffX):
            f, axarr = plt.subplots(3, sharex=False)  #  create subplots
            axarr[0].plot(self.diffX)
            axarr[0].set_ylabel('diffs extrema x indice')

            axarr[1].plot(self.diffAmp)
            axarr[1].set_ylabel('diffs extrema amp')
            if self.threshold:
                axarr[1].axhline(self.threshold, linestyle='--')

            axarr[2].plot(self.diffFusion)
            axarr[2].set_ylabel('diffs fusion')


            if saveFig==True:
                plt.savefig(figFolder+'diff_'+str(figNumber)+'.png')
                plt.close(f)  #  close the subplot close(f), plt.close() is to close plt

        if saveFig==False:
            plt.show()


    def pltRefinedNotePtFc(self, x, y, p, rsquare, polyVar, vibFreq, saveFig=False, figFolder='./', figNumber=0):

        '''
        plot the pitchtrack and the fitting curve
        '''

        if not os.path.exists(figFolder):
            os.makedirs(figFolder)

        #plt.figure()  #  create a new figure
        ######  pitchtrack figure
        f, ax = plt.subplots()
        ax.plot(x, y, 'b.', label='note pitchtrack')
        if len(self.ySmooth) > 3:           # do not print the smooth track if length of ySmooth is too small
            ax.plot(x, self.ySmooth, 'b--', label='smoothed pitchtrack')

        if p is not None:
            pc = np.poly1d(p) # polynomial class
            ax.plot(x, pc(x), 'r-', label='fitting curve')

        if len(self.minimaInd) or len(self.maximaInd):
            ax.plot(x[self.minimaInd], self.ySmooth[self.minimaInd], 'gv', markersize=10.0)  #  markers of local minimas
            ax.plot(x[self.maximaInd], self.ySmooth[self.maximaInd], 'g^', markersize=10.0)  #  markers of local maximas

            #  add text above the extreme
            sortedIndices = sorted(self.minimaInd+self.maximaInd)
            for ii in range(len(sortedIndices)):
                ax.annotate(str(ii), xy = (x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]),
                            xytext=(x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]+0.05))

        # axarr[0].legend(loc='best')
        ax.set_ylabel('normed midinote number')
        # plt.xlabel('frame')
        ax.set_title('rsquare: '+str(rsquare)+'polyVar'+str(polyVar)+'vibratoFreq: '+str(vibFreq))

        if saveFig==True:
            plt.savefig(figFolder+str(figNumber)+'.png')
            plt.close(f)  #  close the subplot close(f), plt.close() is to close plt

        if saveFig==False:
            plt.show()

    def featureExtractionProcess(self,rpt,sbp,curvefittingDeg):

        # feature extraction part used in noteSegmentationFeatureExtraction(self, ...)

        xRpt, yRpt = self.normalizeNotePt(rpt)                               #  normalise x to [0,1], remove y DC
        self.localMinMax(yRpt,sbp)                                           #  local minima and extrema of pitch track
                                                                            #  use the same smooth pts as before
        # p = self.pitchContourFit(xRpt, yRpt, curvefittingDeg)                 #  curve fitting
        # rsquare = self.polyfitRsquare(xRpt, yRpt, p)                         #  polynomial fitting coefs

        p, rsquare, yinterp = self.pitchContourLMBySM(xRpt,yRpt)             #  1 degree curve fitting statsmodels

        polyVar = self.polyfitVariance(xRpt,yinterp,p)                       #  variance of fitting curve
        standardd = np.std(yRpt,dtype=np.float64)                            #  standard deviation
        extAmo = self.extremaAmount()                                        #  extrema amount
        contourLen = len(xRpt)                                              #  contour length
        fcc = self.fittingcurveCrossing(xRpt,yinterp,p)                      #  fitting curve crossing
        vibFreq,residuals = self.vibFreq(rpt, curvefittingDeg)                #  vibrato frequence

        # vibrato coefs by vibrato Nadine
        vibB, vibBExt, vibBFreq, vibBFreqMin, vibBFreqMax = vibrato.vibrato(rpt)
        # print len(rpt), vibB
        vibOut = self.vibratoCoef(rpt,vibB,vibBExt,vibBFreq)

        # featureVec = np.append(p,[rsquare,vibFreq])
        featureVec = np.append(p,[rsquare,polyVar,standardd,extAmo,contourLen,fcc,vibFreq])  #  test different feature vector
        extrema = sorted(self.minimaInd+self.maximaInd)

        return featureVec,extrema,vibOut

    def noteSegmentationFeatureExtraction(self,pitchtrackNoteFolderPath,featureVecFolderPath,pitchtrackNotePredictFolderPath,
                                          recordingNames,segCoef=0.2,predict=False,evaluation=False):

        '''
        This process will do    1) segment pitchtrack into notes which boundaries are given by pYIN
                                2) refined segmentation searching stable part
                                3) calculate features on refined segments
        :param pitchtrackNoteFolderPath:
        :param featureVecFolderPath:
        :param recordingNames:
        :param predict:
        :return: refined segments boundaries, refined segments pitch contours
        '''

        ############################################## segmentation ########################################################
        # below two lines will do segmentaion on calculation the pyin
        # ptSeg = pitchtrackSegByNotes.pitchtrackSegByNotes(samplingFreq, frameSize, hopSize)
        # ptSeg.doSegmentation(pitchtrack, fs.m_oMonoNoteOut)

        ptSeg = pitchtrackSegByNotes.pitchtrackSegByNotes()

        if evaluation:
            evalu1 = evalu.Evaluation()                                                     #  evaluation object
            COnOffall, COnall, OBOnall, OBOffall,gt,st = 0,0,0,0,0,0

        for rn in recordingNames:
            pitchtrack_filename = pitchtrackNoteFolderPath+rn+'_pitchtrack.csv'
            monoNoteOut_filename = pitchtrackNoteFolderPath+rn+'_monoNoteOut.csv'

            ptSeg.doSegmentationForPyinVamp(pitchtrack_filename, monoNoteOut_filename)

            if evaluation:
                coarseSegmentation_filename = pitchtrackNoteFolderPath+rn+'_coarseSeg.txt'
                ptSeg.coarseSegmentation(monoNoteOut_filename,coarseSegmentation_filename)      #  groundtruth segmentation

            # ptSeg.pltNotePitchtrack(saveFig=True, figFolder='../jingjuSegPic/laosheng/train/male_13/pos_3_midinote/')

        ###################### calculate the polynomial fitting coefs and vibrato frequency ############################
        #  use pitch track ptseg.pitchtrackByNotes from last step

            featureDict = {}                                    # feature vectors dictionary
            segmentsExport = {}                                 # refined segmentation boundaries
            refinedPitchcontours = {}                           # refined pitch contours
            extremas = {}                                       # extremas
            vibrato = {}
            curvefittingDeg = 1
            jj = 1
            jjNone = []
            jjj = 1
            for ii in range(len(ptSeg.pitchtrackByNotes)):
                pt = ptSeg.pitchtrackByNotes[ii][0]
                pt = np.array(pt, dtype=np.float32)
                x, y = self.normalizeNotePt(pt)                  #  normalise x to [0,1], remove y DC
                sbp = self.localMinMax(y)                        #  local minima and extrema of pitch track
                self.diffExtrema(x,y)                            #  the amplitude difference of minima and extrema
                self.segmentPointDetection1(segCoef)             #  do the extrema segmentation here

                #  nc1.segmentPointDetection2()  # segmentation point
                self.segmentRefinement(pt)                       #  do the refined segmentation
                #print self.refinedNotePts

                for rpt in self.refinedNotePts:
                    # print jj

                    featureVec,extrema,vibOut = self.featureExtractionProcess(rpt,sbp,curvefittingDeg)
                    if featureVec[0]:
                        refinedPitchcontours[jj] = rpt.tolist()
                        featureDict[jj] = featureVec.tolist()
                        extremas[jj] = extrema
                        vibrato[jj] = vibOut
                    else:
                        jjNone.append(jj)

                    #  this plot step is slow, if we only want the features, we can comment this line
                    #nc1.pltRefinedNotePtFc(xRpt, yRpt, p, rsquare, polyVar, vibFreq, saveFig=True,
                    #                        figFolder='../jingjuSegPic/laosheng/train/refinedSegmentCurvefit/'+rn+'_curvefit_refined/',
                    #                        figNumber = jj)
                    jj += 1

                if predict:
                    # construct the segments frame vector: frame boundary of segments
                    noteStartFrame = ptSeg.noteStartEndFrame[ii][0]
                    noteEndFrame = ptSeg.noteStartEndFrame[ii][1]

                    extremaInd = np.array(self.extrema)
                    segmentsInd = extremaInd[self.segments]+noteStartFrame
                    segmentsInd = np.insert(segmentsInd,0,noteStartFrame)
                    segmentsInd = np.append(segmentsInd,noteEndFrame)+2                     # +2 for sonicVisualizer alignment
                    # segmentsExport[jjj] = str(segmentsInd)
                    for kk in range(len(segmentsInd)-1):
                        if jjj not in jjNone:
                            segmentsExport[jjj] = [segmentsInd[kk],segmentsInd[kk+1]]       #  segmentation boundary
                        jjj += 1

            if evaluation:
                # evaluate segmentation
                COnOff, COn, OBOn, OBOff = \
                    evalu1.coarseEval(ptSeg.coarseSegmentationStartEndFrame,segmentsExport.values())
                COnOffall += COnOff
                COnall += COn
                OBOnall += OBOn
                OBOffall += OBOff
                gt += len(ptSeg.coarseSegmentationStartEndFrame)
                st += len(segmentsExport.values())

            # write feature into json
            featureFilename = featureVecFolderPath+rn+'.json'
            with open(featureFilename, 'w') as outfile:
                json.dump(featureDict, outfile)

            if predict:
                # output segments boundary frames pitch contours
                outJsonDict = {'refinedPitchcontours':refinedPitchcontours,'boundary':segmentsExport,
                               'extremas':extremas,'vibrato':vibrato}
                with open(pitchtrackNotePredictFolderPath+rn+'_refinedSegmentFeatures.json', "w") as outfile:
                    json.dump(outJsonDict,outfile)
                    # for se in segmentsExport:
                    #     # outfile.write(str(int(se[0]))+'\t'+str(se[1])+'\n')
                    #     outfile.write(str(se)+'\n')

        if evaluation:
            # print COnOffall,COnall,OBOnall,OBOffall,gt,st
            COnOffF,COnF,OBOnRateGT,OBOffRateGT = evalu1.metrics(COnOffall,COnall,OBOnall,OBOffall,gt,st)
            return COnOffF,COnF,OBOnRateGT,OBOffRateGT


