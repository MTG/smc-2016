import json
import noteClass as nc
import numpy as np
import copy
import utilFunc as uf
import matplotlib.pyplot as plt
import pitchtrackSegByNotes as ptSeg
import evaluation as evalu

class RefinedSegmentsManipulation(object):
    def __init__(self):
        self.nc1 = nc.noteClass()
        self.ptSeg1 = ptSeg.pitchtrackSegByNotes()
        self.evalu1 = evalu.Evaluation()
        self.evaluationMetrics = []

        self.resetRepresentation()

    def resetRepresentation(self):
        self.representationFeaturesDict = {}
        self.representationBoundariesDict = {}
        self.representationTargetDict = {}
        self.representationSegmentPts = {}

    def mergeExtremas(self, extrema):

        # delete extremas which are too close

        ii = len(extrema)-1
        if ii >= 2:
            while ii > 0:
                if extrema[ii]-extrema[ii-1] == 2:
                    extrema[ii-1] = extrema[ii-1]+1
                    extrema.pop(ii)
                elif extrema[ii]-extrema[ii-1] == 1:
                    extrema.pop(ii)
                ii -= 1


    def eliminateOversegmentation(self,slopeTh = 10.0,pitchTh=2.0,boundaryTh=3.0,flatnoteSlope=20.0):

        # abs(the slope of a curve - the slope of its previous curve) < threshold1
	    # abs(the beginning pitch of a curve - ending pitch of its previous curve) < threshold2
	    # the beginning frame of a curve - ending frame of its previous curve < threshold3

        jj = len(self.representationTargetDict)-1
        keys = sorted(self.representationTargetDict.keys())
        while jj > 0:
            secondKey = keys[jj]
            firstKey = keys[jj-1]
            #  we do this only for linear class
            if self.representationTargetDict[secondKey]==0 and self.representationTargetDict[firstKey]==0:
                slope1 = self.representationFeaturesDict[secondKey][0][0]*self.nc1.samplerate/self.nc1.hopsize
                slope2 = self.representationFeaturesDict[firstKey][0][0]*self.nc1.samplerate/self.nc1.hopsize
                slope1 = np.arctan(slope1)*180/np.pi
                slope2 = np.arctan(slope2)*180/np.pi
                len1 = len(self.representationSegmentPts[secondKey])*self.nc1.hopsize/self.nc1.samplerate
                len2 = len(self.representationSegmentPts[firstKey])*self.nc1.hopsize/self.nc1.samplerate

                # this coefs should be ajusted
                tooshortNote = 0.05

                if (abs(slope1)>flatnoteSlope and abs(slope2)>flatnoteSlope and slope1*slope2>0) or \
                    (abs(slope1)<flatnoteSlope and abs(slope2)<flatnoteSlope) or \
                    (len1<tooshortNote or len2<tooshortNote):
                    # 1) if slope1 and slope2 are all not flat notes, have the same sign
                    # 2) if slope1 and slope2 are all flat notes
                    # 3) if seg1 or seg2 too short

                    slopeDiff = abs(slope1-slope2)
                    pitchDiff = abs(self.representationSegmentPts[secondKey][0]-self.representationSegmentPts[firstKey][-1])
                    boundaryDiff = self.representationBoundariesDict[secondKey][0]-self.representationBoundariesDict[firstKey][-1]

                    if (slopeDiff<slopeTh or (abs(slope1)<flatnoteSlope and abs(slope2)<flatnoteSlope))\
                            and pitchDiff<pitchTh and boundaryDiff<boundaryTh:
                        #  new segment pitch contour
                        previousStartingFrame = self.representationBoundariesDict[firstKey][0]
                        currentStartingFrame = self.representationBoundariesDict[secondKey][0]
                        lenPreviousSegment = len(self.representationSegmentPts[firstKey])
                        previousEndingFrame = previousStartingFrame+lenPreviousSegment
                        lenMissing = int(currentStartingFrame-previousEndingFrame+1)

                        x = [0,lenMissing]
                        y = [self.representationSegmentPts[firstKey][-1],self.representationSegmentPts[secondKey][0]]
                        xinterp = range(1,lenMissing)
                        # interpolation of missing part
                        yinterp = np.interp(xinterp, x, y)
                        yinterp = yinterp.tolist()

                        self.representationSegmentPts[firstKey]=\
                            self.representationSegmentPts[firstKey]+yinterp+self.representationSegmentPts[secondKey]

                        #  linear regression coeffs
                        y = self.representationSegmentPts[firstKey]
                        leny = len(y)
                        x = np.linspace(0,leny-1,leny)
                        p,rsquared,yWithoutOutliers = self.nc1.pitchContourLMBySM(x,y)
                        self.representationFeaturesDict[firstKey] = [p,rsquared]

                        #  boundary
                        self.representationBoundariesDict[firstKey]=\
                            [self.representationBoundariesDict[firstKey][0],self.representationBoundariesDict[secondKey][-1]]

                        self.representationBoundariesDict.pop(secondKey)
                        self.representationFeaturesDict.pop(secondKey)
                        self.representationTargetDict.pop(secondKey)
                        self.representationSegmentPts.pop(secondKey)
            jj -= 1

    def pltRepresentationHelper(self,xboundary,y,representation,pitchHz):

        for jj in range(len(y)):
            if pitchHz:
                yout = uf.midi2pitch(y[jj])
            else:
                yout = y[jj]
            representation.append([xboundary[jj],yout])

        return representation

    def pltRepresentation(self,pitchHz=False,
                          figurePlt=False,
                          figureFilename='test.png',
                          pitchtrackFilename=None,
                          refinedSegFilename=None,
                          evaluation=False):

        if figurePlt:
            plt.figure()
            ii = 0                                              #  a counter for interleave plot the pitch contours

            # plot original pitch track for comparison
            # if pitchtrackFilename:
            #     frameStartingTime, pt = self.ptSeg1.readPyinPitchtrack(pitchtrackFilename)
            #     frameInd = frameStartingTime*self.nc1.samplerate/self.nc1.hopsize
            #     ptmidi = uf.pitch2midi(pt)
            #     plt.plot(frameInd,ptmidi)

        representation = []
        startFrames = []
        endFrames =[]
        sortedKey = sorted(self.representationTargetDict.keys())

        for key in sortedKey:
            leny = len(self.representationSegmentPts[key])
            startFrame = self.representationBoundariesDict[key][0]
            # end frame self.representationBoundariesDict[key][1] is not correct
            endFrame = startFrame + leny
            xboundary = np.linspace(startFrame,endFrame-1,leny)

            startFrames.append(startFrame)
            endFrames.append(endFrame)

            if self.representationTargetDict[key] == 0 or self.representationTargetDict[key] == 1:
                p = self.representationFeaturesDict[key][0]
                if p is not None:
                    x = np.linspace(0,leny-1,leny)
                    pc = np.poly1d(p) # polynomial class
                    y = pc(x)

                    self.pltRepresentationHelper(xboundary,y,representation,pitchHz)

                    if figurePlt:                               #  plot linear curve
                        color = 'k'
                        if ii%2 == 0:
                            color = 'r'
                        plt.plot(xboundary,y,color=color)

                        # angle of the curve, only using in condition first degree regression
                        slope = self.representationFeaturesDict[key][0][0]*self.nc1.samplerate/self.nc1.hopsize
                        slope = np.arctan(slope)*180/np.pi

                        plt.annotate('{0:.2f}'.format(slope), xy = (xboundary[0], y[0]),
                            xytext=(xboundary[0]+0.5, y[0]),fontsize=5)

                        # annotation of boundary
                        #plt.annotate(xboundary[0], xy = (xboundary[0], y[0]),
                        #    xytext=(xboundary[0]+0.5, y[0]),fontsize=5)
                        #plt.annotate(xboundary[-1], xy = (xboundary[-1], y[-1]),
                        #    xytext=(xboundary[-1]+0.5, y[-1]),fontsize=5)
                        ii += 1

            else:
                y = self.representationSegmentPts[key]

                self.pltRepresentationHelper(xboundary,y,representation,pitchHz)

                if figurePlt:                                   #  plot vibrato
                    plt.plot(xboundary,y,'b')

        if figurePlt:                                           #  regulate the figure size
            fig = plt.gcf()
            dpi = 180.0
            fig.set_size_inches(int(len(representation)*2/dpi),10.5)
            fig.savefig(figureFilename, dpi=dpi)
            #print figureFilename, ', plot length: ',len(representation)

        if refinedSegFilename and not evaluation:
            with open(refinedSegFilename, 'w+') as outfile:
                outfile.write('startFrame'+','+'endFrame'+','+'startTime'+','+'endTime'+'\n')
                for ii in range(len(startFrames)):
                    outfile.write(str(int(startFrames[ii]))+
                                  ','+str(int(endFrames[ii]))+
                                  ','+str(startFrames[ii]*self.nc1.hopsize/float(self.nc1.samplerate))+
                                  ','+str(endFrames[ii]*self.nc1.hopsize/float(self.nc1.samplerate))+'\n')

        return representation

    def loopPart(self,keys,segmentPts,targetDict,boundaries,extremas,firstPass = True):
        jj = 1
        for key in keys:
            if not firstPass:                                                               #  no need to order
                jj = key
            y = segmentPts[key]
            if targetDict[key] == 0:                                                        #  linear
                leny = len(y)
                x = np.linspace(0,leny-1,leny)

                if firstPass:
                    p,rsquared,yinterp = self.nc1.pitchContourLMBySM(x,y)
                else:
                    # second time pass, we use 3 degrees linear regression
                    p = self.nc1.pitchContourFit(x, y, 3)
                    rsquared = self.nc1.polyfitRsquare(x, y, p)

                if p is not None:
                    self.representationFeaturesDict[jj] = [p.tolist(),rsquared]              # linear regression coeffs
                    self.representationBoundariesDict[jj] = boundaries[key]
                    self.representationTargetDict[jj] = targetDict[key]
                    self.representationSegmentPts[jj] = y
                jj += 1

            if targetDict[key] == 2:                                                        #  vibrato
                ycopy = copy.copy(y)
                vibRate,residuals = self.nc1.vibFreq(y, 1)
                vibExtent = self.nc1.vibExt(residuals,vibRate)
                vibFreq = np.mean(y)
                features = np.array([vibRate,vibExtent,vibFreq],dtype=np.float64)

                self.representationFeaturesDict[jj] = features.tolist()           # vibrato features
                self.representationBoundariesDict[jj] = boundaries[key]
                self.representationTargetDict[jj] = targetDict[key]
                self.representationSegmentPts[jj] = ycopy
                jj += 1

            if targetDict[key] == 1:                                                        #  others
                extrema = extremas[key]
                extrema.insert(0,0)
                extrema.append(len(y))
                self.mergeExtremas(extrema)

                for ii in range(len(extrema)-1):
                    # split by extrema
                    yseg = y[extrema[ii]:extrema[ii+1]]
                    yboundary = [boundaries[key][0]+extrema[ii],boundaries[key][0]+extrema[ii+1]]

                    leny = len(yseg)
                    xseg = np.linspace(0,leny-1,leny)
                    p,rsquared,yinterp = self.nc1.pitchContourLMBySM(xseg,yseg)

                    self.representationFeaturesDict[jj] = [p.tolist(),rsquared]             # linear regression coeffs
                    self.representationBoundariesDict[jj] = yboundary
                    self.representationTargetDict[jj] = 0                                   #  target becomes linear
                    self.representationSegmentPts[jj] = yseg
                    jj += 1

    def writeRegressionPitchtrack(self,originalPitchtrackFilename,regressionPitchtrackFilename,representation):
        '''
        write regression pitch track
        :param originalPitchtrackFilename:
        :param regressionPitchtrackFilename:
        :param representation:
        :return:
        '''
        if regressionPitchtrackFilename:
            with open(regressionPitchtrackFilename, 'w+') as outfile:
                #  non-voice insertion
                representationNpArray = np.array(representation)
                # maxIndexRepresentation = max(representationNpArray[:,0])
                frameStartingTime, originalPitchtrack = self.ptSeg1.readPyinPitchtrack(originalPitchtrackFilename)
                lenFrame = len(frameStartingTime)
                wholeIndex = range(2,int(lenFrame)+1)
                outfile.write('frame'+','+'time'+','+'pitch'+','+'freq'+','+'noteStr'+'\n') 
                for wi in wholeIndex:
                    if wi in representationNpArray[:,0]:
                        wiIndex = np.where(representationNpArray[:,0]==wi)[0][0]
                        value = representationNpArray[wiIndex,1]                    #  if value is in regression pitchtrack
                   	freq = uf.midi2pitch(value)
			noteStr = uf.cents2pitch(uf.hz2cents(float(freq)))
		    else:
                        value = -100.0                                              #  if value is NOT in regression pitchtrack
                    	freq = -100.0
			noteStr = 'null'
		    outfile.write(str(wi)+','
                                +str(wi*float(self.nc1.hopsize)/self.nc1.samplerate)+','
                                +str(value)+','
				+str(freq)+','
				+noteStr+'\n')

    def process(self, refinedSegmentFeaturesName, targetFilename, representationFilename,figureFilename,
                regressionPitchtrackFilename=None,
                originalPitchtrackFilename=None,
                refinedSegGroundtruthFilename=None,
                evaluation = False,
                slopeTh=10.0,flatnoteTh=20.0):

        # load refined segment features and target class
        with open(refinedSegmentFeaturesName) as data_file:
            refinedSegmentFeatures = json.load(data_file)
        with open(targetFilename) as data_file:
            targetDict = json.load(data_file)

        # separate segment features
        segmentPts = refinedSegmentFeatures['refinedPitchcontours']
        boundaries = refinedSegmentFeatures['boundary']
        extremas = refinedSegmentFeatures['extremas']
        vibrato = refinedSegmentFeatures['vibrato']

        # vibrato adjust by nadine vibrato detection
        for k in targetDict.keys():
            if targetDict[k] == 2:                              #  if it's a vibrato, make it as other
                targetDict[k] = 1
            if vibrato[k][0]:
                targetDict[k] = 2

        # representation features for each segment
        self.resetRepresentation()

        keysInt = [int(i) for i in segmentPts.keys()]
        keys = [str(i) for i in sorted(keysInt)]

        self.loopPart(keys,segmentPts,targetDict,boundaries,extremas,True)          #  estimate segment slope
        #  elimination
        self.eliminateOversegmentation(slopeTh = slopeTh,pitchTh=2.0,boundaryTh=3.0,flatnoteSlope=flatnoteTh)
        #  second time elimination
        self.eliminateOversegmentation(slopeTh = slopeTh,pitchTh=2.0,boundaryTh=3.0,flatnoteSlope=flatnoteTh)

        #  estimate segment LP coefs 3 degrees
        self.loopPart(sorted(self.representationTargetDict.keys()),self.representationSegmentPts,
                      self.representationTargetDict,self.representationBoundariesDict,
                      [],False)

        #  write the boundary and pitch contours in pitch contour file
        #  save refined segmentation boundary: evaluation = False and refinedSegFilename not None
        representation = self.pltRepresentation(pitchHz=False,figurePlt=False,
                                                figureFilename=figureFilename,
                                                pitchtrackFilename=originalPitchtrackFilename,
                                                refinedSegFilename=refinedSegGroundtruthFilename,
                                                evaluation = evaluation)

        # representation[:,1] = uf.midi2pitch(representation[:,1])

        # write the regression pitch track
        self.writeRegressionPitchtrack(originalPitchtrackFilename,regressionPitchtrackFilename,representation)

        outDict = {'Features':self.representationFeaturesDict,'boundary':self.representationBoundariesDict,
                   'target':self.representationTargetDict,'pitchcontours':self.representationSegmentPts}

        # write representation to json
        with open(representationFilename, "w") as outfile:
            json.dump(outDict, outfile)

        # print self.representationTargetDict

        # evaluation
        if evaluation and refinedSegGroundtruthFilename:
            self.ptSeg1.refinedSegmentation(refinedSegGroundtruthFilename)
            COnOffF,COnF,OBOnRateGT,OBOffRateGT = \
                self.evalu1.coarseEval(self.ptSeg1.refinedSegmentationStartEndFrame,
                                       self.representationBoundariesDict.values())
            self.evaluationMetrics = [COnOffF,COnF,OBOnRateGT,OBOffRateGT,
                                      len(self.ptSeg1.refinedSegmentationStartEndFrame),
                                      len(self.representationBoundariesDict.values())]
