'''
calculate the vibrato section, vibrato frequence and extend
'''
import essentia.standard as ess
import numpy as np

def vibrato(pitch):
    sampleRate = 44100/256
    frameSize = int(round(0.5*sampleRate))
    frameSize = frameSize if len(pitch)>=frameSize else len(pitch)      #  dynamic frameSize
    fftSize = 4*frameSize
    dBLobe = 15
    dBSecondLobe = 20
    minFreq = 2.0
    maxFreq = 8.0
    minExt = 30.0
    maxExt = 250.0
    fRef = 55.0
    winAnalysis = 'hann'

    # INIT
    f0  = []
    for f in pitch:
        if f < 0:
            f0.append(0)
        else:
            f0.append(f)

    # GET CONTOUR SEGMENTS
    startC=[]
    endC=[]
    if f0[0]>0:
        startC.append(0)

    for ii in range(len(f0)-1):
        if (abs(f0[ii+1]))>0 and f0[ii]==0:
            startC.append(ii+1)
        if (f0[ii+1]==0 and abs(f0[ii])>0):
            endC.append(ii)

    if len(endC)<len(startC):
        endC.append(len(f0))

    WINDOW = ess.Windowing(type=winAnalysis, size = frameSize, zeroPadding=fftSize-frameSize)
    
    # vibrato annotations
    vibSec = [0]*len(f0)
    vibFreq = [0]*len(f0)
    vibFreqMin = [0]*len(f0)
    vibFreqMax = [0]*len(f0)
    vibExt = [0]*len(f0)

    vibBRaw = []
    vibFreqMinRaw = []
    vibFreqMaxRaw = []

    # ANALYSE EACH SEGMENT
    for ii in range (len(startC)):
        # get segment in cents
        contour = f0[startC[ii]:endC[ii]]
        # contour = 1200*np.log2(np.array(contour)/fRef)            #  it's already in midi
        # frame-wise FFT
        for jj in range(0,len(contour)-frameSize,int(round(frameSize/2))):
            frame = contour[jj:jj+frameSize]
            extend = max(frame)-min(frame)
            minFrame = min(frame)
            maxFrame = max(frame)
            frame = frame-np.mean(frame)
            frame = ess.MovingAverage(size=5)(frame)
            frame = WINDOW(frame)

            # extent constraint
            #if extend<minExt or extend>maxExt:
            #    continue

            S = ess.Spectrum(size = fftSize)(frame)

            #fig = plt.figure()
            #ax = fig.add_subplot(211)

            #plt.plot(S)

            locs, amp = ess.PeakDetection(maxPeaks = 3, orderBy = 'amplitude')(S)
            freqs=locs*(fftSize/2+1)*sampleRate/fftSize
            #print freqs, amp
            if len(freqs)<=0:
                continue
            if freqs[0]<minFreq or freqs[0]>maxFreq: # strongest peak is not in considered range
                continue
            if len(freqs)>1: # there is a second peak
                if freqs[1]>minFreq and freqs[1]<maxFreq:
                    continue
                if 20*np.log10(amp[0]/amp[1])<dBLobe:
                    continue
            if len(freqs)>2: #there is a third peak
                if freqs[2]>minFreq and freqs[2]<maxFreq: # it is also in the vibrato range
                    continue
                if 20*np.log10(amp[0]/amp[2])<dBSecondLobe:
                    continue
            vibSec[startC[ii]+jj:startC[ii]+jj+frameSize] = f0[startC[ii]+jj:startC[ii]+jj+frameSize]
            vibExt[startC[ii]+jj:startC[ii]+jj+frameSize] = [extend]*frameSize
            vibFreq[startC[ii]+jj:startC[ii]+jj+frameSize] = [freqs[0]]*frameSize
            vibFreqMin[startC[ii]+jj:startC[ii]+jj+frameSize] = [minFrame]*frameSize
            vibFreqMax[startC[ii]+jj:startC[ii]+jj+frameSize] = [maxFrame]*frameSize

            vibBRaw.append((startC[ii]+jj, startC[ii]+jj+frameSize))
            vibFreqMinRaw.append(minFrame)
            vibFreqMaxRaw.append(maxFrame)
    
    # section filter            
    toRemove = vibratoSectionFilter(vibBRaw, vibFreqMinRaw, vibFreqMaxRaw)
    for tr in toRemove:
        start = vibBRaw[tr][0]
        end = vibBRaw[tr][1]-frameSize/2
        vibSec[start:end] = [0]*(frameSize/2)
        vibExt[start:end] = [0]*(frameSize/2)
        vibFreq[start:end] = [0]*(frameSize/2)
        vibFreqMin[start:end] = [0]*(frameSize/2)
        vibFreqMax[start:end] = [0]*(frameSize/2)

    # write vibrato sections boundary
    vibB = []
    vibBExt = []
    vibBFreq = []
    vibBFreqMin = []
    vibBFreqMax = []
    if vibSec[0]>0:
        vibB.append(0)
    for ii in range(len(f0)-1):
        if abs(vibSec[ii+1])>0 and vibSec[ii]==0:
            vibB.append(ii+1)
        if vibSec[ii+1]==0 and vibSec[ii]>0:
            vibB.append(ii)
    if vibSec[-1]>0:
        endC.append(len(f0)-1)

    assert (len(vibB)%2==0), "vib boundary should be even!"

    for ii in range(len(vibB)/2):
        vibBExt.append(vibExt[vibB[2*ii]:vibB[2*ii+1]+1])
        vibBFreq.append(vibFreq[vibB[2*ii]:vibB[2*ii+1]+1])
        vibBFreqMin.append(vibFreqMin[vibB[2*ii]:vibB[2*ii+1]+1])
        vibBFreqMax.append(vibFreqMax[vibB[2*ii]:vibB[2*ii+1]+1])
    return vibB, vibBExt, vibBFreq, vibBFreqMin, vibBFreqMax

def vibratoSectionFilter(vibBRaw, vibFreqMinRaw, vibFreqMaxRaw):
    '''
    filter out inner sections segment
    '''
    flag = [0]
    jj = 0
    for ii in range(1,len(vibBRaw)):
        if vibBRaw[ii][0] < vibBRaw[ii-1][1]:
            flag.append(jj)
        else:
            jj += 1
            flag.append(jj)
    
    # vibrato frequency minimum and maximum in group
    vibFreqMinSection = []
    vibFreqMaxSection = []
    jj = 0
    tempMin = []
    tempMax = []
    for ii in range(len(vibFreqMinRaw)):
        if flag[ii] != jj:
            jj = flag[ii]
            vibFreqMinSection.append([ii-len(tempMin), tempMin])
            vibFreqMaxSection.append([ii-len(tempMax), tempMax])
            tempMin = [vibFreqMinRaw[ii]]
            tempMax = [vibFreqMaxRaw[ii]]
        else:
            tempMin.append(vibFreqMinRaw[ii])
            tempMax.append(vibFreqMaxRaw[ii])

    # filter out section
    toRemove = []
    for ii in range(len(vibFreqMinSection)):
        minSection = vibFreqMinSection[ii][1]
        maxSection = vibFreqMaxSection[ii][1]
   
        if len(minSection) >=3:
            meanMinSection = np.mean(minSection)
            stdMinSection = np.std(minSection)
            meanMaxSection = np.mean(maxSection)
            stdMaxSection = np.std(maxSection)
            
            for jj in range(len(minSection)):
                if (minSection[jj] > (meanMinSection + 2*stdMinSection) or minSection[jj] < (meanMinSection - 2*stdMinSection)) or (maxSection[jj] > (meanMaxSection + 2*stdMaxSection) or maxSection[jj] < (meanMaxSection - 2*stdMaxSection)):
                    toRemove.append(vibFreqMinSection[ii][0]+jj)
            #print minSection, meanMinSection, stdMinSection
            #print maxSection, meanMaxSection, stdMaxSection
    return toRemove

def vibratoFilter(vibB, vibBExt, vibBFreq, vibBFreqMin, vibBFreqMax):
    '''
    filter out the vibrato length less than one period
    '''
    hoptime = 256/44100.0
    vibLen = []
    for ii in range(len(vibB)/2):
        startB = vibB[2*ii]
        endB = vibB[2*ii+1]
        vibLen.append((endB-startB+1)*hoptime)

    vibBMeanPeriod = []
    for vibBF in vibBFreq:
        period = 2.0/np.mean(vibBF)
        vibBMeanPeriod.append(period)

    vibBFreqMinStd = []
    for vibBFM in vibBFreqMin:
        vibBFreqMinStd.append(np.std(vibBFM))
    #print vibBFreqMinStd

    vibBFreqMaxStd = []
    for vibBFM in vibBFreqMax:
        vibBFreqMaxStd.append(np.std(vibBFM))
    #print vibBFreqMaxStd

    toRemove = []
    for ii in range(len(vibLen)):
        if vibLen[ii] < vibBMeanPeriod[ii] or min(vibBFreqMinStd[ii], vibBFreqMaxStd[ii]) > 50:
            toRemove.append(ii)
    
    toRemove = np.array(toRemove)
    vibB = np.delete(vibB, np.hstack((2*toRemove, 2*toRemove+1)))
    vibBExt = np.delete(vibBExt, toRemove)
    vibBFreq = np.delete(vibBFreq, toRemove)
    vibBFreqMinStd = np.delete(vibBFreqMinStd, toRemove)

    return vibB, vibBExt, vibBFreq
