// Global variables
var currTala = $("#taal").val();
var isBlinking = false;
var beatsPT = {'teen':16, 'jhap':10, 'ek': 12, 'rupak':7};
var beatDuration =60.0/parseFloat($("#tempoControl").val());
var samaDuration = beatDuration*beatsPT[currTala];
var pulsePeriod;
var samaScheduler;
var beatScheduler;
var nextSamaTime;

var currTempo = Number($("#tempoControl").val());
var imgSama = document.getElementById('samaImage');
imgSama.style.visibility = 'hidden';
var imgBeat = document.getElementById('beatImage');
imgBeat.style.visibility = 'hidden';

function onTempoChange() {
	console.log($("#tempoControl").val());
	currTempo = Number($("#tempoControl").val());
        $.ajax({
                    type: "POST",
                    url: "http://127.0.0.1:5000/set_tempo",
                    data: JSON.stringify({'tempo': 60.0/parseFloat($("#tempoControl").val())}),
                    contentType: 'application/json',
                    dataType: 'json'
                }).done(function(data) {
                    console.log(data);
                });
        var currStr = $("#tempoControl").val();
        console.log(currStr.concat(" beats(matras) per minute"))
        $("#tempoValDisp").html(currStr.concat(" beats per minute"))
        stopMetronome();
        startMetronome();
                
}


function onTaalChange(){
	currTala = $("#taal").val();
        $.ajax({
                    type: "POST",
                    url: "http://127.0.0.1:5000/set_taal",
                    data: JSON.stringify({'taal': $("#taal").val()}),
                    contentType: 'application/json',
                    dataType: 'json'
                }).done(function(data) {
                    console.log(data);
                });
                console.log($("#taal").val());
                stopMetronome();
                startMetronome();
}

function getPulsePeriod(){
    return 60.0/parseFloat($("#tempoControl").val());
    
}
function getBlinkOnSpeed() {
    
	return (60 * 1000) / (currTempo);
}

function getBlinkOffSpeed() {
	return (60 * 1000) / (currTempo);
}

function samaArrived(){
    nextSamaTime = new Date().getTime() + samaDuration * 1000;  
    $("#samaText").text("Sama");
    imgSama.style.visibility = 'visible';
    setTimeout(function (){$("#samaText").text("    ");}, 300);
    setTimeout(function (){imgSama.style.visibility = 'hidden';}, 500);
}

function beatArrived(){
    $("#beatText").text("Beat");
    imgBeat.style.visibility = 'visible';
    setTimeout(function (){$("#beatText").text("");}, 200);
    setTimeout(function (){imgBeat.style.visibility = 'hidden';}, 200);
}


function startMetronome() {
        
        pulsePeriod = getPulsePeriod();
        samaDuration = pulsePeriod*beatsPT[currTala];
        beatDuration = pulsePeriod;
        
        buildMetronometrack();
        playMetronomeAudio();
        
        nextSamaTime = new Date().getTime() + samaDuration * 1000;
        
        samaScheduler = setInterval(samaArrived, samaDuration*1000);
        beatScheduler = setInterval(beatArrived, beatDuration*1000);
    
        $("#recordButton").removeAttr("disabled");
        $("#recordingInfo").html("Ready to record ...");
}
function stopMetronome(){
    
    stopMetronomeAudio();
    clearInterval(samaScheduler);
    clearInterval(beatScheduler);
    $("#recordButton").attr("disabled", "disabled");
    $("#recordingInfo").html("Start Metronome to begin!");
    
}

function offBlinker() {
	var $disp = $("#tempoView");
	$disp.html("");

	var $blinker = $("#blinker");
	$blinker.css("background-color","white");

	setTimeout(onBlinker, getBlinkOffSpeed());
}