import numpy as np
import os,sys

# add src patt
dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)

import concatenateSegment as cs
import dtwRong
import dtwSankalp
import plottingCode as pc
import alignment as align

# definition
segmentFileFolder = './segmentFiles/'
outputFileFolder = './outputFiles/'

teacherTitle = 'weiguojia_section_pro'
studentTitle = 'weiguojia_section_amateur'

teacherMonoNoteOutFilename = segmentFileFolder+teacherTitle+'_monoNoteOut_midi.txt'
studentMonoNoteOutFilename = segmentFileFolder+studentTitle+'_monoNoteOut_midi.txt'

teacherRepresentationFilename = segmentFileFolder+teacherTitle+'_representation.json'
studentRepresentationFilename = segmentFileFolder+studentTitle+'_representation.json'

teacherNoteAlignedFilename = outputFileFolder+teacherTitle+'_noteAligned.csv'
studentNoteAlignedFilename = outputFileFolder+studentTitle+'_noteAligned.csv'

teacherSegAlignedFilename = outputFileFolder+teacherTitle+'_segAligned.csv'
studentSegAlignedFilename = outputFileFolder+studentTitle+'_segAligned.csv'

cs1 = cs.ConcatenateSegment()
align1 = align.Alignment()

#################################################### note alignment ####################################################

# read note file
noteStartingTime_t, noteDurTime_t, midiNote_t = cs1.readPyinMonoNoteOutMidi(teacherMonoNoteOutFilename)
noteStartingTime_s, noteDurTime_s, midiNote_s = cs1.readPyinMonoNoteOutMidi(studentMonoNoteOutFilename)
noteStartingFrame_t, noteEndingFrame_t = cs1.getNoteFrameBoundary(noteStartingTime_t, noteDurTime_t)
noteStartingFrame_s, noteEndingFrame_s = cs1.getNoteFrameBoundary(noteStartingTime_s, noteDurTime_s)

# get concatenated pitch track
notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t = \
    cs1.generateNotePitchtrack(noteStartingFrame_t, noteEndingFrame_t, midiNote_t)
notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s = \
    cs1.generateNotePitchtrack(noteStartingFrame_s, noteEndingFrame_s, midiNote_s)

# resampling note
if len(notePts_t) > len(notePts_s):
    notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s \
        = cs1.resampling(notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s, len(notePts_t))
elif len(notePts_s) > len(notePts_t):
    notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t \
        = cs1.resampling(notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t, len(notePts_s))

# alignment
path = dtwSankalp.dtw1d_generic(notePts_t,notePts_s)
path_t = path[0]
path_s = path[1]

# get path index for each note
noteStartingFramePath_t, noteEndingFramePath_t \
    = align1.getPathIndex(path_t,noteStartingFrameConcatenate_t,noteEndingFrameConcatenate_t)
noteStartingFramePath_s, noteEndingFramePath_s \
    = align1.getPathIndex(path_s,noteStartingFrameConcatenate_s,noteEndingFrameConcatenate_s)

# alignment
alignedNote_t = align1.alignment2(noteStartingFramePath_t, noteEndingFramePath_t,
                            noteStartingFramePath_s, noteEndingFramePath_s)

alignedNote_s = align1.alignment2(noteStartingFramePath_s, noteEndingFramePath_s,
                            noteStartingFramePath_t, noteEndingFramePath_t)

# print noteStartingFramePath_t, noteEndingFramePath_t
# print noteStartingFramePath_s, noteEndingFramePath_s

# print alignedNote_t, alignedNote_s


############################################ segmentation alignment ####################################################

segmentPts_t,boundaries_t,target_t = cs1.readRepresentation(teacherRepresentationFilename)
segmentPts_s,boundaries_s,target_s = cs1.readRepresentation(studentRepresentationFilename)

concatenatePts_t,segStartingFrame_t,segEndingFrame_t = cs1.concatenate(segmentPts_t)
concatenatePts_s,segStartingFrame_s,segEndingFrame_s = cs1.concatenate(segmentPts_s)

# resampling note
if len(notePts_t) > len(notePts_s):
    concatenatePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s \
        = cs1.resampling(concatenatePts_s, segStartingFrame_s, segEndingFrame_s, len(concatenatePts_t))
elif len(notePts_s) > len(notePts_t):
    concatenatePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t \
        = cs1.resampling(concatenatePts_t, segStartingFrame_t, segEndingFrame_t, len(concatenatePts_s))

# do dtw
#dist, D, path = dtwRong.dtw(concatenatePts_t,concatenatePts_s)
path = dtwSankalp.dtw1d_generic(concatenatePts_t,concatenatePts_s)
path_t = path[0]
path_s = path[1]

# alignment
# frameInd_t = align1.path2frameInd(path_t,concatenateMap_t)
# pathIndSegment_t = align1.indexSegmentation(frameInd_t, boundaries_t)
#
# frameInd_s = align1.path2frameInd(path_s,concatenateMap_s)
# pathIndSegment_s = align1.indexSegmentation(frameInd_s, boundaries_s)
#
# alignDict_t = align1.alignment(pathIndSegment_t,pathIndSegment_s)
# alignDict_s = align1.alignment(pathIndSegment_s,pathIndSegment_t)
#
# # plotting
# pc.plotAll(segmentPts_t,boundaries_t,target_t,alignDict_t,
#             segmentPts_s,boundaries_s,target_s,alignDict_s)
# print alignDict_t
# print alignDict_s

# get path index for each note
segStartingFramePath_t, segEndingFramePath_t \
    = align1.getPathIndex(path_t,segStartingFrame_t,segEndingFrame_t)
segStartingFramePath_s, segEndingFramePath_s \
    = align1.getPathIndex(path_s,segStartingFrame_s,segEndingFrame_s)

# alignment
alignedSeg_t = align1.alignment2(segStartingFramePath_t, segEndingFramePath_t,
                                segStartingFramePath_s, segEndingFramePath_s)

alignedSeg_s = align1.alignment2(segStartingFramePath_s, segEndingFramePath_s,
                                segStartingFramePath_t, segEndingFramePath_t)

# print alignedSeg_t, alignedSeg_s

############################################ save aligned file #########################################################

def stral(al):
    out = ''
    for c in al:
        out = out+str(c)+' '
    return out

with open(teacherNoteAlignedFilename, 'w+') as outfile:
    for al in alignedNote_t:
        outfile.write(str(al[0])+','+stral(al[1])+'\n')

with open(studentNoteAlignedFilename, 'w+') as outfile:
    for al in alignedNote_s:
        outfile.write(str(al[0])+','+stral(al[1])+'\n')

with open(teacherSegAlignedFilename, 'w+') as outfile:
    for al in alignedSeg_t:
        outfile.write(str(al[0])+','+stral(al[1])+'\n')

with open(studentSegAlignedFilename, 'w+') as outfile:
    for al in alignedSeg_s:
        outfile.write(str(al[0])+','+stral(al[1])+'\n')
