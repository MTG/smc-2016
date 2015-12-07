# Beijing opera singing intonation analysis and education tool


The online tool can be accessed at [https://dunya.compmusic.upf.edu/jingju-icassp-2016/](https://dunya.compmusic.upf.edu/jingju-icassp-2016/)

Music Technology Group, Universitat Pompeu Fabra, Barcelona  
rong.gong@upf.edu, yile.yang01@estudiant.upf.edu  
[http://compmusic.upf.edu](http://compmusic.upf.edu)

## Summary

This is the demonstration of a singing intonation analysis 
and education tool specially developed for phrase-level Beijing 
opera singing training. It lies within the domain 
“Audio and Acoustic Signal Processing”. The functionality of 
this tool is to analyse the singing intonation of the student 
and reveal interactively his/her problem on both note level 
and F0 contour’s segment level by the comparison 
with the teacher. The technologies used in this tool 
contain F0 extraction, note transcription, F0 contour segmentation 
and note/segment alignment. F0 contour segmentation developed in 
Beijing opera context enables us to analyse the subtle details of 
the intonation. The note/segment alignment facilitate the student 
to check the corresponding notes or segments and then find the 
intonation problem on a finer scale. 
The web page implementation of this tool 
offers a great potential for self-learning and MOOC application.

## Beijing opera singing training

In professional Beijing opera singing training, the teacher has 
absolute authority. The teacher's singing is seen as the standard 
to which the student's imitation should be as close 
as possible. The training process is on phrase-level. The teacher 
sings firstly one short phrase which is usually less than 8 syllables, 
then the student imitates this phrase. Finally, the teacher will 
give comments or suggestions from the perspectives of intonation, 
rhythm, timbre, loudness andphonation. This training process 
is shown in the example video below.

## Algorithm

From the audio recording, the algorithm extracts 
the F0 pitch contour and transcribes it into notes. The F0 pitch 
contour is then segmented into ascending, descending, flat and vibrato
parts for the purpose of finer analysis. The notes and segments are 
finally aligned between teacher and student.

## Resources

1. pYIN, a pitch tracking and note transcription algorithm is used to extract the F0 pitch contour and transcribe the notes.
2. Essentia, an open-source C++ library for audio analysis and audio-based music information retrieval is used to extract vibrato.
3. A global constraint DTW algorithm is used to align the notes and the segments.

## Credits

This project uses following software

1. [d3.js](http://d3js.org/)
2. [wavesurfer.js](http://wavesurfer-js.org/)
3. [Recorder.js](https://github.com/mattdiamond/Recorderjs)





