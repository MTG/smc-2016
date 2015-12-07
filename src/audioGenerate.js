//global variables
var audio_context;
var metVolNode;
var metronomeVolValue = 0.8;
var isPlaying = false;
var isMetronomePlaying = false;
var source1;
var nChannels = 1;
var tabla_strokes;
var audioBuffer;
var audioBufferMetronome;
var sampleRate;
var theka = 'teental';
var total_out_dur;
var tala_info;
var playSound;
var bufferTailLen = 2.0;
// var beatPosition = {'teen': {'durratio': [0, .25, .5, .75], 'bol': ['hiClick', 'lowClick', 'lowClick', 'lowClick']},
//     'ek': {'durratio': [0, .25, .5, .75], 'bol': ['hiClick', 'lowClick', 'lowClick', 'lowClick']},
//     'jhap': {'durratio': [0, 0.2, .5, .7], 'bol': ['hiClick', 'lowClick', 'lowClick', 'lowClick']},
//     'rupak': {'durratio': [0, .428, .714], 'bol': ['hiClick', 'lowClick', 'lowClick']}
// };
// var clickSounds;
// var metronome;

// function setBarLength(length){
//     barLength = length
// }

// function setTheka(theka_inp){
//     theka = theka_inp;
// }

// var getClicks = new XMLHttpRequest();
// getClicks.open("GET", "http://127.0.0.1:5000/get_click_sounds", true);
// getClicks.send();
// getClicks.onreadystatechange = function() {
//     if (getClicks.readyState == 4 && getClicks.status == 200) {
//         clickSounds = JSON.parse(getClicks.responseText);
//         startMetronome();
//         onTempoChange();
//         onTaalChange();
//     }
// }

// fetching dictionary of sounds
// var getSound = new XMLHttpRequest();
// getSound.open("GET", "http://127.0.0.1:5000/get_tabla_sounds", true);
// getSound.send();
// getSound.onreadystatechange = function() {
//     if (getSound.readyState == 4 && getSound.status == 200) {
//         tabla_strokes = JSON.parse(getSound.responseText);
//     }
// }

//fetch tala information
// var getTala = new XMLHttpRequest();
// getTala.open("GET", "http://127.0.0.1:5000/get_tala_info", true);
// getTala.send();
// getTala.onreadystatechange = function() {
//     if (getTala.readyState == 4 && getTala.status == 200) {
//         tala_info = JSON.parse(getTala.responseText);
//     }
// }


// function playBackWithDelay() {
//     console.log("Inside playBackWithDelay");
//     $("#recordingInfo").html("Waiting for next sam (downbeat) to play response...");
//     buildTheka();
//     playTheka();
//     setTimeout(startPlayTheka, nextSamaTime - new Date().getTime());
// }

// function playThekaButton() {
//     playTheka();
//     startPlayTheka();
// }
// function buildTheka(){
//     total_out_dur = tala_info.strokeTime[tala_info.strokeTime.length-1] + bufferTailLen;
//     var frameCount = sampleRate*total_out_dur;
//     audioBuffer = audio_context.createBuffer(nChannels, frameCount, sampleRate);
//     for (var channel = 0 ; channel < nChannels; channel ++){
//         var nowBuffering = audioBuffer.getChannelData(channel);
    
//         var len_stroke;
//         for (var bol_ind in tala_info.strokeTime){
//             start = Math.floor(tala_info.strokeTime[bol_ind]*sampleRate);
//             len_stroke = tabla_strokes[tala_info.strokeList[bol_ind]].length
//             for (var ii = 0; ii < len_stroke; ii++){
//                 nowBuffering[start + ii] = nowBuffering[start + ii] + tala_info.strokeAmp[bol_ind]*tabla_strokes[tala_info.strokeList[bol_ind]][ii]/32767.0;
//             }
//         }
//     }
// }

// function buildMetronometrack(){
//     var frameCount = sampleRate*samaDuration;
//     audioBufferMetronome = audio_context.createBuffer(nChannels, frameCount, sampleRate);
//     for (var channel = 0 ; channel < nChannels; channel ++){
//         var nowBuffering = audioBufferMetronome.getChannelData(channel);
    
//         var len_stroke;
//         var barLen = samaDuration;
//         for (var bol_ind in beatPosition[currTala]['bol']){
//             start = Math.floor(samaDuration*beatPosition[currTala]['durratio'][bol_ind]*sampleRate);
//             //console.log(beatPosition[currTala]['durratio'][bol_ind], currTala);
//             len_stroke = clickSounds[beatPosition[currTala]['bol'][bol_ind]].length
//             for (var ii = 0; ii < len_stroke; ii++){
//                 nowBuffering[start + ii] = nowBuffering[start + ii] + clickSounds[beatPosition[currTala]['bol'][bol_ind]][ii]/32767.0;           
//             }
//         }
//         //console.log(nowBuffering);
//     }
// }

// function playMetronomeAudio(){
//     if (isMetronomePlaying == true){
//             stopMetronome();
//     }
//     metronome = audio_context.createBufferSource();
//     metronome.buffer = audioBufferMetronome;
//     metVolNode = audio_context.createGain();
//     metronome.connect(metVolNode);
//     metVolNode.connect(audio_context.destination);
//     metVolNode.gain.value = metronomeVolValue;
//     metronome.loop = true;
//     metronome.start(0);
//     isMetronomePlaying = true;
// };

// function onMetVolChange() {
//     metronomeVolValue = 0.01*parseFloat($("#metVol").val());
//     console.log(metronomeVolValue);
//     metVolNode.gain.value = metronomeVolValue;
// }

// function playTheka(){
//     console.log('Started the playback')
//     if (isPlaying == true){
//             stopPlaying();
//     }
//     playSound = audio_context.createBufferSource();
//     playSound.buffer = audioBuffer;
//     playSound.connect(audio_context.destination);
// };

// function startPlayTheka() {
//     $("#recordingInfo").html("Playing response...");
//     console.log(playSound.buffer.duration)
//     playSound.start(0);  
//     isPlaying = true;
//     setTimeout(function(){$("#recordingInfo").html("Ready to record ...");}, (playSound.buffer.duration - bufferTailLen)*1000);
// }

// function stopPlaying(){
//     if (isPlaying == true){
//         playSound.stop();
//         isPlaying = false;        
//     }
//     if (isMetronomePlaying == true){ 
//         $("#recordingInfo").html("Ready to record ...");
//     }
//     else {
//         $("#recordingInfo").html("Start Metronome to begin!");
//     }
    
// };

// function stopMetronomeAudio(){
//     if (isMetronomePlaying == true){
//         metronome.stop();
//         isMetronomePlaying = false;        
//     }
    
// };

// Function that initializes the audio context
function init() {
    try {
      window.AudioContext = window.AudioContext || window.webkitAudioContext;
      navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
      window.URL = window.URL || window.webkitURL;
      
      audio_context = new AudioContext();
      
    } catch (e) {
      alert('No web audio support in this browser!');
    }
    sampleRate = audio_context.sampleRate;
};





