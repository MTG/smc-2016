import os

from jingjuSegAlign.src import concatenateSegment as cs
from jingjuSegAlign.src import dtwSankalp
from jingjuSegAlign.src import alignment as align


def jingjuSegmentAlignment(phraseNumber):
    # definition
    exampleFolder = os.path.join(dir,'../../data/exampleFiles/'+phraseNumber+'/')
    updateFolder = os.path.join(dir,'../../data/updateFiles/')

    teacherTitle = 'teacher'
    studentTitle = 'student'

    teacherMonoNoteOutFilename = exampleFolder+teacherTitle+'_monoNoteOut.csv'
    studentMonoNoteOutFilename = updateFolder+studentTitle+'_monoNoteOut.csv'

    teacherRepresentationFilename = exampleFolder+teacherTitle+'_representation.json'
    studentRepresentationFilename = updateFolder+studentTitle+'_representation.json'

    # output file name

    teacherNoteAlignedFilename = updateFolder+teacherTitle+'_noteAligned.csv'
    studentNoteAlignedFilename = updateFolder+studentTitle+'_noteAligned.csv'

    teacherSegAlignedFilename = updateFolder+teacherTitle+'_segAligned.csv'
    studentSegAlignedFilename = updateFolder+studentTitle+'_segAligned.csv'

    cs1 = cs.ConcatenateSegment()
    align1 = align.Alignment()

    #################################################### note alignment ####################################################

    # read note file
    noteStartingTime_t, noteDurTime_t, midiNote_t = cs1.readPyinMonoNoteOut(teacherMonoNoteOutFilename)
    noteStartingTime_s, noteDurTime_s, midiNote_s = cs1.readPyinMonoNoteOut(studentMonoNoteOutFilename)
    print 'read monoNoteOut done!'
    noteStartingFrame_t, noteEndingFrame_t = cs1.getNoteFrameBoundary(noteStartingTime_t, noteDurTime_t)
    noteStartingFrame_s, noteEndingFrame_s = cs1.getNoteFrameBoundary(noteStartingTime_s, noteDurTime_s)
    print 'getNoteFrameBoundary done'

    # get concatenated pitch track
    notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t = \
        cs1.generateNotePitchtrack(noteStartingFrame_t, noteEndingFrame_t, midiNote_t)
    notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s = \
        cs1.generateNotePitchtrack(noteStartingFrame_s, noteEndingFrame_s, midiNote_s)
    print 'generateNotePitchtrack done'

    # normalization pitch track
    notePts_t = cs1.pitchtrackNormalization(notePts_t)
    notePts_s = cs1.pitchtrackNormalization(notePts_s)
    print 'normalization done'

    # resampling note
    if len(notePts_t) > len(notePts_s):
        notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s \
            = cs1.resampling(notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s, len(notePts_t))
    elif len(notePts_s) > len(notePts_t):
        notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t \
            = cs1.resampling(notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t, len(notePts_s))
    print 'resampling done'

    print len(notePts_t)
    print len(notePts_s)

    # alignment
    path = dtwSankalp.dtw1d_generic(notePts_t,notePts_s)
    path_t = path[0]
    path_s = path[1]
    print 'dtw done'

    # print notePts_t
    # print notePts_s

    # print path_t, len(path_t), len(notePts_t)
    # print noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t
    # print path_s, len(path_s), len(notePts_s)
    # print noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s

    # get path index for each note
    noteStartingFramePath_t, noteEndingFramePath_t \
        = align1.getPathIndex(path_t,noteStartingFrameConcatenate_t,noteEndingFrameConcatenate_t)
    noteStartingFramePath_s, noteEndingFramePath_s \
        = align1.getPathIndex(path_s,noteStartingFrameConcatenate_s,noteEndingFrameConcatenate_s)
    print 'getPathIndex done'


    # alignment
    alignedNote_t = align1.alignment2(noteStartingFramePath_t, noteEndingFramePath_t,
                                noteStartingFramePath_s, noteEndingFramePath_s)

    alignedNote_s = align1.alignment2(noteStartingFramePath_s, noteEndingFramePath_s,
                                noteStartingFramePath_t, noteEndingFramePath_t)

    print 'alignment done'

    # print noteStartingFramePath_t, noteEndingFramePath_t
    # print noteStartingFramePath_s, noteEndingFramePath_s

    # print alignedNote_t, alignedNote_s


    ############################################ segmentation alignment ####################################################

    segmentPts_t,boundaries_t,target_t = cs1.readRepresentation(teacherRepresentationFilename)
    segmentPts_s,boundaries_s,target_s = cs1.readRepresentation(studentRepresentationFilename)

    concatenatePts_t,segStartingFrame_t,segEndingFrame_t = cs1.concatenate(segmentPts_t)
    concatenatePts_s,segStartingFrame_s,segEndingFrame_s = cs1.concatenate(segmentPts_s)

    concatenatePts_t = cs1.pitchtrackNormalization(concatenatePts_t)
    concatenatePts_s = cs1.pitchtrackNormalization(concatenatePts_s)

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
        for ii, c in enumerate(al):
            if ii != len(al)-1:
                out = out+str(c)+' '
            else:
                out = out+str(c)
        return out

    with open(teacherNoteAlignedFilename, 'w+') as outfile:
        outfile.write('teacher'+','+'student'+'\n')
        for al in alignedNote_t:
	    alignStr = stral(al[1]) if al[1] else 'null'
            outfile.write(str(al[0])+','+alignStr+'\n')

    with open(studentNoteAlignedFilename, 'w+') as outfile:
        outfile.write('student'+','+'teacher'+'\n')
        for al in alignedNote_s:
	    alignStr = stral(al[1]) if al[1] else 'null'
            outfile.write(str(al[0])+','+alignStr+'\n')

    with open(teacherSegAlignedFilename, 'w+') as outfile:
        outfile.write('teacher'+','+'student'+'\n')
        for al in alignedSeg_t:
	    alignStr = stral(al[1]) if al[1] else 'null'
            outfile.write(str(al[0])+','+alignStr+'\n')

    with open(studentSegAlignedFilename, 'w+') as outfile:
        outfile.write('student'+','+'teacher'+'\n')
        for al in alignedSeg_s:
	    alignStr = stral(al[1]) if al[1] else 'null'
            outfile.write(str(al[0])+','+alignStr+'\n')
