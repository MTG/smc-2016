## Table analysis and synthesis module for the HAMR 2015 ISMIR hack

import os
import essentia as es
import essentia.standard as ess
import numpy as np
import pickle
import glob
import utilFunctions as UF
import scipy.spatial.distance as DS
import matplotlib.pyplot as plt

import parameters as params
import csv

rms=ess.RMS()
window = ess.Windowing(type = "hamming")
spec = ess.Spectrum(size=params.Nfft)
zz = np.zeros((params.zeropadLen,), dtype = 'float32')
genmfcc = ess.MFCC(highFrequencyBound = 22000.0, inputSize = params.Nfft/2+1, sampleRate = params.Fs)
hps = ess.HighPass(cutoffFrequency = 240.0)
onsets = ess.Onsets()

strokeLabels = ['dha', 'dhen', 'dhi', 'dun', 'ge', 'kat', 'ke', 'na', 'ne', 're', 'tak', 'te', 'tit', 'tun']

taals = {"teen": {"nmatra": 16, "accents": np.array([4, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1])}, 
         "ek": {"nmatra": 12, "accents": np.array([4, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1])},
         "jhap": {"nmatra": 10, "accents": np.array([4, 1, 2, 1, 1, 3, 1, 2, 1, 1])},
         "rupak": {"nmatra": 7, "accents": np.array([2, 1, 1, 3, 1, 3, 1])}
         }


rolls = [{"bol": ['dha/dha_02', 'te/te_05', 're/re_04', 'dha/dha_02'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])},
         {"bol": ['te/te_02', 're/re_05', 'ke/ke_04', 'te/te_02'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])},
         {"bol": ['ge/ge_02', 'ge/ge_05', 'te/te_04', 'te/te_02'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])},
         {"bol": ['ge/ge_02', 'ge/ge_05', 'dhi/dhi_04', 'na/na_02'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])},
         {"bol": ['dha/dha_02', 'dha/dha_02', 'te/te_05', 'te/te_06'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])},
         {"bol": ['dha/dha_02', 'dha/dha_02', 'dhi/dhi_05', 'na/na_06'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])},
         {"bol": ['na/na_02', 'ge/ge_05', 'te/te_04', 'te/te_02'], "dur": np.array([1.0, 1.0, 1, 1]), "amp": np.array([1.0, 1.0, 1.0, 1.0])}
         ]

# Need different tihais for different letters remaining, start from say,  matras

fullTihais = {"teen_1": {"totDur": 17, "bol": ['te/te_08', 're/re_01', 'ke/ke_01', 'te/te_02', 'dhi/dhi_01', 're/re_01', 'ke/ke_01', 'te/te_02', 'dhi/dhi_01', 'ke/ke_01', 'te/te_02', 'dhi/dhi_01', 'te/te_02', 'dhi/dhi_02'], "dur": np.array([1.0, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1]), "amp": np.array([0.7, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 1, 0.8, 1])},
            "ek_1": {"totDur": 12, "bol": ['dha/dha_01', 'ge/ge_01', 'dhi/dhi_01', 'dha/dha_01', 'ge/ge_01', 'dhi/dhi_01', 'dha/dha_01', 'ge/ge_01', 'dhi/dhi_01'], "dur": np.array([1.0, 1, 2, 1, 1, 2, 1, 1, 2]), "amp": np.array([0.8, 0.8, 1, 0.8, 0.8, 1, 0.8, 0.8, 1])},
            "jhap_1": {"totDur": 11, "bol": ['dha/dha_01', 'ge/ge_01', 'dha/dha_01', 'dha/dha_02', 'ge/ge_01', 'dha/dha_01', 'dha/dha_02', 'dha/dha_03', 'ge/ge_01'], "dur": np.array([1.0, 2, 1, 1, 2, 1, 1, 1, 1]), "amp": np.array([1.0, 0.8, 1, 1, 0.8, 1, 0.8, 0.8, 1])},
            "rupak_1": {"totDur": 14, "bol": ['te/te_08', 're/re_01', 'ke/ke_01', 'te/te_02', 'dhi/dhi_01', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_02', 'dhi/dhi_01', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_02'], "dur": np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), "amp": np.array([0.8, 0.8, 0.8, 0.8, 1, 0.8, 0.8, 0.8, 0.8, 1, 0.8, 0.8, 0.8, 0.8])}
            }

thekaSlow = {"teen": {"totDur": 16, "bol": ['dha/dha_01', 'dhi/dhi_01', 'dhi/dhi_02', 'dha/dha_01', 'dha/dha_01', 'dhi/dhi_01', 'dhi/dhi_02', 'dha/dha_01', 'dha/dha_01', 'tun/tun_01', 'tun/tun_02', 'na/na_01', 'dhi/dhi_01', 'na/na_01', 'dha/dha_01', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_08', 'dha/dha_02', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_02'], "dur": np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25]), "amp": np.array([1.0, 0.8, 0.8, 0.8, 1, 0.8, 0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 1, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7, 0.7]),},
             "ek": {"totDur": 12, "bol": ['dhi/dhi_01', 'dhi/dhi_02', 'na/na_01', 'kat/kat_01', 'tun/tun_01', 'na/na_01', 'dhi/dhi_02', 'na/na_01', 'dha/dha_01', 'ge/ge_01', 'ke/ke_01', 'te/te_08', 'dha/dha_01', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_08', 'dha/dha_02', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_02'], "dur": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25]), "amp": np.array([1.0, 0.8, 0.8, 1, 0.8, 0.8, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 1, 0.7, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 0.7])},
             "jhap": {"totDur": 10, "bol": ['dhi/dhi_01', 'na/na_01', 'dhi/dhi_02', 'dhi/dhi_02', 'na/na_01', 'tun/tun_01', 'na/na_01', 'dha/dha_01', 'ke/ke_01', 'te/te_08', 'dha/dha_02', 'ke/ke_01', 'te/te_02'], "dur": np.array([1.0, 1, 1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25]), "amp": np.array([1, 0.8, 0.9, 0.9, 0.8, 1, 0.8, 0.9, 0.7, 0.7, 0.9, 0.7, 0.7])},
             "rupak": {"totDur": 7, "bol": ['tun/tun_01', 'tun/tun_01', 'te/te_08', 'ke/ke_01', 'dhi/dhi_01', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_08', 'dhi/dhi_02', 'te/te_08', 're/re_01', 'ke/ke_01', 'te/te_08'], "dur": np.array([1.0, 1, 0.5, 0.5, 1, 0.25, 0.25, 0.25, 0.25, 1, 0.25, 0.25, 0.25, 0.25]), "amp": np.array([0.9, 0.9, 0.9, 0.9, 1, 0.7, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 0.7])},
             }

dataPath = '../dataset/44k/'

def getFeatSequence(inputFile,pulsePos):
    if type(inputFile) == str:
        audio = ess.MonoLoader(filename = inputFile, sampleRate = params.Fs)()
    else:
        audio = inputFile           # Assuming mono 
    frameCounter = 0
    pool = es.Pool()
    pool.add('samples',audio)
    for frame in ess.FrameGenerator(audio, frameSize = params.frmSize, hopSize = params.hop):
        ts = params.hop/params.Fs*frameCounter + params.frmSize/float(2*params.Fs)
        zpFrame = np.hstack((frame,zz))
        mag = spec(window(zpFrame))
        mfccBands,mfccSeq = genmfcc(mag)
        pool.add('rms',rms(mag))
        pool.add('mfcc',mfccSeq)
        pool.add('time',ts)
        frameCounter += 1
    prmsVal = pool['rms']/np.max(pool['rms'])
    pool.remove('rms')
    pool.add('rms', prmsVal)
    if pulsePos != None:
        pulsePos = np.append(pulsePos,len(audio)/params.Fs)
        for tp in xrange(len(pulsePos)-1):
            pool.add('pst', pulsePos[tp])
            pool.add('pet', pulsePos[tp+1])
            temp1 = np.where(pool['time'] >= pulsePos[tp])[0]
            temp2 = np.where(pool['time'] < pulsePos[tp+1])[0]
            binIndices = np.intersect1d(temp1, temp2)
            pool.add('pmfcc', np.mean(pool['mfcc'][binIndices,:], axis = 0))
            pool.add('prms', np.mean(pool['rms'][0][binIndices]))
    else:
        pool.add('pst', 0.0)
        pool.add('pet', len(audio)/params.Fs)
        pool.add('pmfcc', np.mean(pool['mfcc'], axis = 0))
        pool.add('prms', np.mean(pool['rms'][0], axis = 0))
    return pool

def buildStrokeModels(strokeLabels, dataBasePath):
    poolFeats = {}
    poolFeats = []
    print "Building stroke models..."
    for stroke in strokeLabels:
        print stroke
        filenames = glob.glob(dataBasePath + stroke + os.sep + '*.wav')
        for fpath in filenames:
            fname = os.path.split(fpath)[1].rsplit('.')[0]
            feat = {'strokeId': [stroke + os.sep + fname], 'feat': getFeatSequence(inputFile = fpath, pulsePos = None)}
            poolFeats.append(feat) 
    return poolFeats
def getPulsePosFromAnn(inputFile):
    pulsePos = np.array([])
    with open(inputFile,'rt') as csvfile:
        sreader = csv.reader(csvfile)
        for row in sreader:
            pulsePos = np.append(pulsePos,float(row[0]))
    return pulsePos

def InitSystem():
    poolFeats = buildStrokeModels(strokeLabels, dataPath)
    
strokeModelsG = buildStrokeModels(strokeLabels, dataPath)

def genAudioFromStrokeSeq(strokeModels,strokeSeq,strokeAmp,strokeTime):
    # Generates a numpy array sequence of values from fnames at the given times
    tail = np.max(np.diff(strokeTime))+1  # Assuming 60 second to be the smallest tempo
    audio = np.zeros(int(np.round((strokeTime[-1] + tail)*params.Fs)))
    print len(audio)
    tsSamp = np.round(strokeTime*params.Fs)
    print tsSamp
    for k in range(len(strokeSeq)):
        chooseInd = [x for x in range(len(strokeModels)) if strokeModels[x]['strokeId'][0] == strokeSeq[k]]
        print chooseInd, strokeSeq[k]
        strokeAudio = strokeModels[chooseInd[0]]['feat']['samples'][0]        
        lenAudio = len(strokeAudio)
        print tsSamp[k], lenAudio, tsSamp[k]+lenAudio
        audio[tsSamp[k]:tsSamp[k]+lenAudio] = audio[tsSamp[k]:tsSamp[k]+lenAudio] + strokeAmp[k]*strokeAudio
    # The last sample
    return audio[:tsSamp[-1]+lenAudio]

def getInvCovarianceMatrix(poolFeats):
    dataMat = np.zeros((len(params.selectInd),len(poolFeats)))
    # dataMat = np.array([])
    for k in range(len(poolFeats)):
        dataMat[:,k] = np.array(poolFeats[k]['feat']['pmfcc'][0][params.selectInd])
    invC = np.linalg.inv(np.cov(dataMat))
    return invC

invCmat = getInvCovarianceMatrix(strokeModelsG)

def genSimilarComposition(pulsePeriod, taalInfo, pieceDur, iAudio, strokeModels = strokeModelsG, invC = None):
    if strokeModels == None:
        strokeSeq = None
        strokeTime = None 
        opulsePos = None
    else:
        iPos = np.arange(0.0,pieceDur,pulsePeriod)
        testFeatFull = getFeatSequence(iAudio,iPos)
        feat_rms = es.array([ testFeatFull['rms'][0] ])
        onsets_rms = onsets(feat_rms, [1 ])
        # plt.plot(testFeatFull['time'], testFeatFull['rms'][0])
        # print onsets_rms, testFeatFull['pst']
        #plt.stem(testFeatFull['pst'],np.max(testFeatFull['rms'][0])*np.ones(len(testFeatFull['pst'])),'r')
        # plt.stem(onsets_rms,np.max(testFeatFull['rms'][0])*np.ones(len(onsets_rms)),'g')        
        # plt.show()
        testFeat = testFeatFull['pmfcc']
        # print testFeat.shape
        Npulse = testFeat.shape[0]
        Ndata = len(strokeModels)
        strokeSeq = []
        strokeTime = np.array([])
        strokeAmp = np.array([])
        tscurr = 0.0
        mPos = 1
        mPosMax = taals[taalInfo]["nmatra"]
        # opulsePos = np.arange(0,pieceDur,pulsePeriod)
        # take off with a theka
        for k in xrange(len(thekaSlow[taalInfo]['bol'])):
            strokeTime = np.append(strokeTime,tscurr)
            tscurr = tscurr + thekaSlow[taalInfo]['dur'][k]*pulsePeriod
            strokeSeq.append(thekaSlow[taalInfo]['bol'][k])
            strokeAmp = np.append(strokeAmp, thekaSlow[taalInfo]['amp'][k])
        # Then get feature based improv
        for k in range(Npulse):
            # First find the best timbrally similar stroke through a very basic distance
            ftIn = testFeat[k,params.selectInd]
            distVal = 1e6*np.ones(Ndata)
            for p in range(Ndata):
                ftOut = strokeModels[p]['feat']['pmfcc'][0][params.selectInd]
                distVal[p] = DS.mahalanobis(ftIn,ftOut,invC)
            # Now find how many onsets exist in the "pulse"
            ind1 = np.where(onsets_rms >= testFeatFull['pst'][k])[0]
            ind2 = np.where(onsets_rms < testFeatFull['pet'][k])[0]
            indSel = np.intersect1d(ind1,ind2)
            print mPos
            if len(indSel) == 0:
                if mPos == 1:
                    chooseInd = [x for x in range(len(strokeModels)) if ('dha' in strokeModels[x]['strokeId'][0] or 'dhi' in strokeModels[x]['strokeId'][0])]
                    chooseInd = np.random.permutation(len(chooseInd))[0]
                    strokeSeq.append(strokeModels[chooseInd]['strokeId'][0])
                    strokeAmp = np.append(strokeAmp, 1)
                    strokeTime = np.append(strokeTime,tscurr)
                    tscurr = tscurr + pulsePeriod
                else:
                    # Choose a base stroke
                    chooseInd = [x for x in range(len(strokeModels)) if ('ge' in strokeModels[x]['strokeId'][0])]
                    chooseInd = np.random.permutation(len(chooseInd))[0]
                    strokeSeq.append(strokeModels[chooseInd]['strokeId'][0])
                    strokeAmp = np.append(strokeAmp, 0.8 + 0.1*np.random.rand())
                    strokeTime = np.append(strokeTime,tscurr)
                    tscurr = tscurr + pulsePeriod
            elif len(indSel) == 1:
                # One onset only
                onsets_pulse = onsets_rms[indSel] - testFeatFull['pst'][k]
                # If this is sam, then add some strong beat here, like a dha or dhi
                if mPos == 1:
                    chooseInd = [x for x in range(len(strokeModels)) if ('dha' in strokeModels[x]['strokeId'][0] or 'dhi' in strokeModels[x]['strokeId'][0])]
                    chooseInd = np.random.permutation(len(chooseInd))[0]
                    strokeSeq.append(strokeModels[chooseInd]['strokeId'][0])
                    strokeAmp = np.append(strokeAmp, 1)
                    strokeTime = np.append(strokeTime,tscurr)
                    tscurr = tscurr + pulsePeriod
                else:
                    if np.random.rand() > params.strokeDoublingP:
                        strokeSeq.append(strokeModels[np.argmin(distVal)]['strokeId'][0])
                        strokeAmp = np.append(strokeAmp, 0.8 + 0.1*np.random.rand())
                        strokeTime = np.append(strokeTime,tscurr)
                        tscurr = tscurr + pulsePeriod
                    else:
                        # Choose one more randomly from a ringing stroke
                        chooseInd = [x for x in range(len(strokeModels)) if ('tun' in strokeModels[x]['strokeId'][0] or 'na' in strokeModels[x]['strokeId'][0] or 'dun' in strokeModels[x]['strokeId'][0] )]
                        chooseInd = np.random.permutation(len(chooseInd))[0]
                        strokeSeq.append(strokeModels[chooseInd]['strokeId'][0])
                        strokeAmp = np.append(strokeAmp, 0.8 + 0.1*np.random.rand())
                        strokeTime = np.append(strokeTime,tscurr)
                        tscurr = tscurr + pulsePeriod/2
                        # And get the timbrally similar one also
                        strokeSeq.append(strokeModels[np.argmin(distVal)]['strokeId'][0])
                        strokeAmp = np.append(strokeAmp, 0.7 + 0.1*np.random.rand())
                        strokeTime = np.append(strokeTime,tscurr)
                        tscurr = tscurr + pulsePeriod/2
            else:
                if mPos == 1:
                    # Choose a dha or dhi for half duration
                    chooseInd = [x for x in range(len(strokeModels)) if ('dha' in strokeModels[x]['strokeId'][0] or 'dhi' in strokeModels[x]['strokeId'][0])]
                    chooseInd = np.random.permutation(len(chooseInd))[0]
                    strokeSeq.append(strokeModels[chooseInd]['strokeId'][0])
                    strokeAmp = np.append(strokeAmp, 1)
                    strokeTime = np.append(strokeTime,tscurr)
                    tscurr = tscurr + pulsePeriod/2
                    # Now choose half the roll
                    chooseInd = np.random.permutation(len(rolls))[0]
                    rollNow = rolls[chooseInd]
                    for k in xrange(2):
                        strokeSeq.append(rollNow['bol'][k])
                        strokeTime = np.append(strokeTime,tscurr)
                        tscurr = tscurr + pulsePeriod/4
                        strokeAmp = np.append(strokeAmp, rollNow['amp'][k])
                else:
                    # Play a roll otherwise!
                    chooseInd = np.random.permutation(len(rolls))[0]
                    rollNow = rolls[chooseInd]
                    if np.random.rand() > params.rollP:
                        nStrokesRoll = 2
                    else:
                        nStrokesRoll = 4
                    # Now get the roll     
                    for k in range(nStrokesRoll):
                        strokeSeq.append(rollNow['bol'][k])
                        strokeTime = np.append(strokeTime,tscurr)
                        tscurr = tscurr + pulsePeriod/nStrokesRoll
                        strokeAmp = np.append(strokeAmp, rollNow['amp'][k])
                        
            # Increment metrical position 
            mPos = mPos + 1
            if mPos > mPosMax:
                mPos = 1
                
        # Put filler strokes till the cycle ends
        while mPos <= mPosMax:
            # One random stroke
            chooseInd = np.random.permutation(len(strokeModels))
            strokeSeq.append(strokeModels[chooseInd[0]]['strokeId'][0])
            strokeAmp = np.append(strokeAmp, 0.9 + 0.1*np.random.rand())
            strokeTime = np.append(strokeTime,tscurr)
            tscurr = tscurr + pulsePeriod/2.0
            # Another random stroke
            strokeSeq.append(strokeModels[chooseInd[1]]['strokeId'][0])
            strokeAmp = np.append(strokeAmp, 0.9 + 0.1*np.random.rand())
            strokeTime = np.append(strokeTime,tscurr)
            tscurr = tscurr + pulsePeriod/2.0
            mPos = mPos + 1
        
        # Finish with a tihai
        for k in xrange(len(fullTihais[taalInfo + '_1']['bol'])):
            chooseInd = [x for x in range(len(strokeModels)) if strokeModels[x]['strokeId'][0] == fullTihais[taalInfo + '_1']['bol'][k]]
            strokeSeq.append(strokeModels[chooseInd[0]]['strokeId'][0])
            strokeTime = np.append(strokeTime,tscurr)
            tscurr = tscurr + fullTihais[taalInfo + '_1']['dur'][k]*pulsePeriod
            strokeAmp = np.append(strokeAmp, fullTihais[taalInfo + '_1']['amp'][k])
    return testFeatFull, strokeSeq, strokeTime, strokeAmp, pulsePeriod # np.median(np.diff(opulsePos))

def getJawaabLive(ipAudio, ipulsePer, iTaal = "teen", strokeModels = strokeModelsG):
    # If poolFeats are not built, give an error!
    if strokeModels == None:
        print "Train models first before calling getJawaab() ..."
        strokeModels = InitSystem()
    else:
        print "Getting jawaab..."
        print iTaal
        pulsePeriod = ipulsePer
        print pulsePeriod
        if type(ipAudio) == str:
            audioIn = ess.MonoLoader(filename = ipAudio, sampleRate = params.Fs)()
            audioIn = hps(audioIn)
        else:
            audioIn = hps(ipAudio)
        fss = params.Fs
        testFeatFull, strokeSeq, strokeTime, strokeAmp, opulsePer = genSimilarComposition(pulsePeriod, iTaal, pieceDur = len(audioIn)/params.Fs, iAudio = audioIn, strokeModels = strokeModels, invC = invCmat)
        #audioOut = genAudioFromStrokeSeq(strokeModels, strokeSeq, strokeAmp, strokeTime)
        #plt.plot(audioOut)
        #plt.show()
    return testFeatFull, strokeSeq, strokeTime, strokeAmp, opulsePer

def testModuleLive(inputFile = '../dataset/testInputs/testInput_3.wav', pulsePos = getPulsePosFromAnn('../dataset/testInputs/testInput_3.csv')):    
    global strokeModelsG
    ipulsePer = np.median(np.diff(pulsePos))/10
    # print ipulsePer
    fss, ipAudio = UF.wavread(inputFile)
    print "Analysing input..."
    testFeatFull, strokeSeq, strokeTime, strokeAmp, opulsePer = getJawaabLive(ipAudio, ipulsePer)
    audioOut = genAudioFromStrokeSeq(strokeModelsG,strokeSeq,strokeAmp,strokeTime)
    return testFeatFull, audioOut, strokeSeq, strokeTime, strokeAmp, opulsePer
    
if __name__ == "__main__":
    print "Testing..."
    testModuleLive()
    # print "Stored output file to %s" %outFile
    