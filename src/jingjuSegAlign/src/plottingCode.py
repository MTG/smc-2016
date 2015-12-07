import matplotlib.pyplot as plt
import numpy as np

def plotPitchtrackSegments(pts1,bd1,tg1,ad1,ax):
    sortedKey = sorted([int(k) for k in pts1.keys()])
    ii = 0
    jj = 0
    annotation = []
    for key in range(0,len(pts1)):
        annotation.append(str(key)+':'+str(ad1[key]))
    for key in sortedKey:
        key = str(key)
        leny = len(pts1[key])
        startFrame = bd1[key][0]
        # end frame self.representationBoundariesDict[key][1] is not correct
        endFrame = startFrame + leny
        xboundary = np.linspace(startFrame,endFrame-1,leny)

        if tg1[key]==0 or tg1[key]==1:
            color = 'k'
            if ii%2 == 0:
                color = 'r'
            ax.plot(xboundary,pts1[key],color=color)
            ii += 1

        if tg1[key]==2:
            ax.plot(xboundary,pts1[key],'b')

        ax.annotate(annotation[jj], xy = (xboundary[0], pts1[key][0]),
                    xytext=(xboundary[0], pts1[key][0]+0.5),fontsize=5)

        jj += 1


def plotAll(pts1,bd1,tg1,ad1,pts2,bd2,tg2,ad2):
    f, axarr = plt.subplots(2, sharex=True)
    plotPitchtrackSegments(pts1,bd1,tg1,ad1,axarr[0])
    plotPitchtrackSegments(pts2,bd2,tg2,ad2,axarr[1])
    plt.show()

