import numpy as np

class Alignment(object):

    def getPathIndex(self,path,noteStartingFrame,noteEndingFrame):
        '''
        get the path index of each note
        :param path:
        :param noteStartingFrame:
        :param noteEndingFrame:
        :return:
        '''

        noteStartingFramePath = []
        noteEndingFramePath = []

        for ii in range(len(noteStartingFrame)):
            npi = []
            for jj in range(0,len(path)):
                if noteStartingFrame[ii] <= path[jj] <= noteEndingFrame[ii]:
                    npi.append(jj)

            if len(npi) < 2:
                s,e = None,None
            else:
                s,e = npi[0],npi[-1]

            noteStartingFramePath.append(s)
            noteEndingFramePath.append(e)

        return noteStartingFramePath, noteEndingFramePath

    def alignment2(self, noteStartingFramePath_t, noteEndingFramePath_t,
                   noteStartingFramePath_s, noteEndingFramePath_s):
        '''

        :param noteStartingFramePath_t:
        :param noteEndingFramePath_t:
        :param noteStartingFramePath_s:
        :param noteEndingFramePath_s:
        :return:
        '''

        th = 0.5
        aligned = []
        for ii in range(len(noteStartingFramePath_t)):
            aligned_ii = []
            s_t = noteStartingFramePath_t[ii]
            e_t = noteEndingFramePath_t[ii]

            # the case if None
            if not s_t or not e_t:
                aligned.append([ii,aligned_ii])
                continue

            for jj in range(len(noteStartingFramePath_s)):
                s_s = noteStartingFramePath_s[jj]
                e_s = noteEndingFramePath_s[jj]

                # the case of None
                if not s_s or not e_s:
                    continue

                if s_t <= s_s and e_t >= e_s:
                    intersection = e_s-s_s
                elif s_t >= s_s and e_t <= e_s:
                    intersection = e_t-s_t
                elif s_t >= s_s and e_t >= e_s:
                    intersection = e_s-s_t
                elif s_t <= s_s and e_t <= e_s:
                    intersection = e_t-s_s
                elif s_s > e_t:
                    break


                if intersection >= (e_s-s_s)*th or intersection >= (e_t-s_t)*th:
                    aligned_ii.append(jj)

            aligned.append([ii,aligned_ii])

        return aligned

    ####################################### deprecated alignment code ##################################################
    def path2frameInd(self,path,map):
        '''
        :param path:
        :param map:
        :return: frame ind of path before concatenation
        '''
        frameInd = []
        for p in path:
            for m in map:
                if m[0] == p:
                    frameInd.append(m[1])
                    break

        return frameInd

    def indexSegmentation(self, frameInd, boundaries):

        pathIndSegment = []
        keys = sorted([int(x) for x in boundaries.keys()])
        for k in keys:
            ks = str(k)
            bd = boundaries[ks]

            fi = []
            for ii in range(len(frameInd)):
                if frameInd[ii]>=bd[0] and frameInd[ii]<bd[1]:
                    fi.append(ii)

            pathIndSegment.append(fi)

        return pathIndSegment

    def alignment(self,pis1,pis2):
        '''
        alignment teacher segment to student segment
        :param pis1:
        :param pis2:
        :return:
        '''

        alignDict = {}
        for ii in range(len(pis1)):
            alignDict[ii] = []
            for p1 in pis1[ii]:
                for jj in range(len(pis2)):
                    if p1 in pis2[jj] and jj not in alignDict[ii]:
                        alignDict[ii].append(jj)

        return alignDict