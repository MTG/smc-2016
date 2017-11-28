(function(window) {
	var app = angular.module("hamrSJ", []);

	// Controller for recorder
	app.controller("SJController", function() {
		this.appName = "jingju-icassp-2016";
	});

	app.controller("RecordController", function($scope,$timeout) {
		this.recording = false;

		////////////////////////////////////////////////////////////////////////////

		// setting up the filesystem api
		//window.requestFileSystem = 

		// seeting up the for the recorder
		var navigator = window.navigator;
		navigator.getUserMedia = (
			navigator.getUserMedia ||
	  		navigator.webkitGetUserMedia ||
	    	navigator.mozGetUserMedia ||
	    	navigator.msGetUserMedia
			);
		var Context = window.AudioContext || window.webkitAudioContext;
		var context = new Context();

		// we need these variables for later use with the stop function
		var mediaStream;
		var rec;

		// Move playback pointer while recording
	    var movePlaybackPointer = function  (buffers) {
		var currentLength = buffers[0].length;
		var currentTime = currentLength / context.sampleRate;
		// console.log(currentTime);
		// console.log(currentTime+"11111");

		d3.select("#playbackPointer_t")
		.attr("x1", x_t(currentTime))
		.attr("y1", height)
		.attr("x2", x_t(currentTime))
		.attr("y2", 0)
		.style("stroke", "black");
		}
	
		var record =function() {

            $("#recordingInfo").html("Recording ...");
            
            console.log("Started Recording!");
            // console.log(new Date().getTime());
            // ask for permission and start recording
		  	navigator.getUserMedia({audio: true}, function(localMediaStream){
		    mediaStream = localMediaStream;

		    // create a stream source to pass to Recorder.js
		    var mediaStreamSource = context.createMediaStreamSource(localMediaStream);

		    // create new instance of Recorder.js using the mediaStreamSource
		    rec = new Recorder(mediaStreamSource, {
		      // pass the path to recorderWorker.js file here
		      workerPath: './thirdparty/recorderWorker.js'
		    },
		    // buffer will input as parameter to movePlaybackPointer 
		    movePlaybackPointer);

		    // start recording
		    rec.record();
		    document.getElementById("recordingInfo").innerHTML = "Recording...  <img src=\"./images/recording.gif\" alt=\"Recording... \" style=\"width:50px;height:14px;\">";

		  }, function(err){
		    console.log('Browser not supported');
		  });
		};

		var stopRec = function() {
		  // stop the media stream
		  // mediaStream.stop();	// deprecated and removed
		  mediaStream.getTracks().forEach(function (track) {
		  	track.stop();
		  });

		  // stop Recorder.js
		  rec.stop();

		  document.getElementById("recordingInfo").innerHTML = "Computing...  <img src=\"./images/computing.gif\" alt=\"Computing... \" style=\"width:50px;height:14px;\">";
                

		  // export it to WAV
		  rec.exportWAV(function(e){
		    rec.clear();

		    var samples = float2int16(e)
	    	var buffer = pcm2mp3(samples,1,44100)
	    	var audioBlob = new Blob(buffer, {type: 'audio/mp3'});
	    	var phraseNumberBlob = new Blob([phraseNumber], {type: 'text/plain'});

			var fd = new FormData();
            fd.append('data', audioBlob);
            fd.append('phraseNumber', phraseNumberBlob);

            $.ajax({
                type: "POST",
                url: "api/do_seg",
                data: fd,
                processData: false,
                contentType: false
            }).done(function(data) {
                console.log("Got the callback!!");
                var data_root = data['data_root'].substr(1);
                // Update data visualization

                noteAlign_T2S_update = data_root + '/teacher_noteAligned.csv ';
                noteAlign_S2T_update = data_root + '/student_noteAligned.csv';
                pitchSeg_s_update = data_root + '/student_refinedSeg.csv';
                pitchSegAlign_T2S_update = data_root + '/teacher_segAligned.csv';
				pitchSegAlign_S2T_update = data_root + '/student_segAligned.csv';
				pitchtrack_s_update = data_root + '/student_regression_pitchtrack.csv';
				noteSeg_s_update = data_root + '/student_monoNoteOut_midi.csv';
				audio_s_update = data_root + '/student.mp3';

	        	updateData(pitchtrack_t_update, pitchtrack_s_update, noteSeg_t_update, noteSeg_s_update, noteAlign_T2S_update, noteAlign_S2T_update, audio_t_update, audio_s_update, pitchSeg_t_update, pitchSeg_s_update, pitchSegAlign_T2S_update, pitchSegAlign_S2T_update);
        		// Update recordingInfo
                document.getElementById("recordingInfo").innerHTML = "Computing finished. Click and hold to record again.";

        		// playBackWithDelay();
        		// console.log(seg_info);
            });
            //console.log(e.slice(0));
		    //Recorder.forceDownload(e, "filename.wav");
		  });

		};


		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// functions for the template
		this.isRecording = function() {
			return this.recording;
		};

		this.startRecording = function() {
			console.log("Recording ....");
			this.recording = true;
			//$("#recordingInfo").html("Waiting for next sam (downbeat) to start recording...");

			$timeout(record, 0);
		};

		this.stopRecording = function(callback) {
			console.log("Recording Stopped ....");
			this.recording = false;
			stopRec();

           
		};
	});

	app.controller("TempoController", function() {
		this.tempo = 100;
	});

})(window);

function float2int16(audioData) {

  // float to int 16 conversion for pcm -> mp3

  var samples = new Int16Array(audioData.length);
  for (var i = 0; i < audioData.length; i++) {
    f = audioData[i] * 32768;
    if( f > 32767 ) f = 32767;
    if( f < -32768 ) f = -32768;
      samples[i] = f;
  }
  return samples
}

function pcm2mp3(samples, channels, sampleRate) {

  // convert samples int16 pcm to mp3

  var liblame = new lamejs();
  var buffer = [];
  mp3enc = new liblame.Mp3Encoder(channels, sampleRate, 128);
  var remaining = samples.length;
  var maxSamples = 1152;
  for (var i = 0; remaining >= maxSamples; i += maxSamples) {
      var mono = samples.subarray(i, i + maxSamples);
      var mp3buf = mp3enc.encodeBuffer(mono);
      if (mp3buf.length > 0) {
          buffer.push(new Int8Array(mp3buf));
      }
      remaining -= maxSamples;
  }
  var d = mp3enc.flush();
  if(d.length > 0){
      buffer.push(new Int8Array(d));
  }
  // console.log('done encoding, size=', buffer.length);

  return buffer
}

function checkTime() {
    console.log("check time", nextSamaTime - new Date().getTime(), new Date().getTime());
}
