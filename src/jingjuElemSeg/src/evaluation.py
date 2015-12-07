'''
This is the code for evaluation.

the coarse evaluation: groundtruth is corrected on the pYIN algorithm by adding stable part segmentation.
the fine evaluation: groundtruth is the vibrato, flat, ascending and descending segmentation which is not based on pYIN segmentation.

'''
import noteClass as nc

class Evaluation(object):

    def __init__(self):
        self.nc1 = nc.noteClass()

    def precision(self,c,groundtruth):
        return c/float(groundtruth)

    def recall(self,c,seg):
        return c/float(seg)

    def Fmeasure(self,p,r):
        return 2*(p*r)/(p+r)

    def errorRateGT(self,e,groundtruth):
        return e/float(groundtruth)

    def errorRateTR(self,e,seg):
        return e/float(seg)

    def errorRatio(self,rateGT,rateTR):
        if not rateGT:
            return None
        else:
            return rateTR/rateGT

    def offsetTh(self,groundtruth):

        thOffset = []
        timeStep = self.nc1.hopsize/self.nc1.samplerate

        for gt in groundtruth:
            twentyP = (gt[1] - gt[0])*timeStep*0.2
            thOffset.append(max(twentyP,0.05))

        return thOffset

    def coarseEval(self, groundtruth, seg):
        '''
        :param groundtruth: a list [[start0,end0],[start1,end1],[start2,end2],...]
        :param seg: segementation, a list [[start0,end0],[start1,end1],[start2,end2],...]
        :return: COnOff, COn, OBOn, OBOff
        '''

        COnOff = 0              # correct onset offset
        COn = 0                 # correct onset
        OBOn = 0                # only incorrect onset
        OBOff = 0               # only incorrect offset

        timeStep = self.nc1.hopsize/self.nc1.samplerate
        thOnset = 0.05          # 50 ms
        thOffset = self.offsetTh(groundtruth)

        for s in seg:
            for gti in range(len(groundtruth)):
                # COn
                if abs(groundtruth[gti][0]-s[0])*timeStep < thOnset:
                    COn += 1
                    # COnOff
                    if abs(groundtruth[gti][1]-s[1])*timeStep < thOffset[gti]:
                        COnOff += 1
                    else:
                        OBOff += 1
                    break       # break the first loop if s have been already mapped to

                if abs(groundtruth[gti][1]-s[1])*timeStep < thOffset[gti] and \
                    abs(groundtruth[gti][0]-s[0])*timeStep >= thOnset:
                    OBOn += 1
                    break

        return COnOff, COn, OBOn, OBOff
        #print groundtruth, seg


    def metrics(self,COnOff, COn, OBOn, OBOff,groundtruthLen,segLen):

        COnOffP = self.precision(COnOff,groundtruthLen)
        COnOffR = self.recall(COnOff,segLen)
        COnOffF = self.Fmeasure(COnOffP,COnOffR)

        COnP = self.precision(COn,groundtruthLen)
        COnR = self.recall(COn,segLen)
        COnF = self.Fmeasure(COnP,COnR)

        OBOnRateGT = self.errorRateGT(OBOn,groundtruthLen)
        OBOnRateTR = self.errorRateTR(OBOn,segLen)
        OBOnRatio = self.errorRatio(OBOnRateGT,OBOnRateTR)

        OBOffRateGT = self.errorRateGT(OBOff,groundtruthLen)
        OBOffRateTR = self.errorRateTR(OBOff,segLen)
        OBOffRatio = self.errorRatio(OBOffRateGT,OBOffRateTR)

        return COnOffF,COnF,OBOnRateGT,OBOffRateGT




