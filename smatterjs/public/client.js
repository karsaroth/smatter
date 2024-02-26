//================= CONFIG =================
// Global Variables
const websocket_uri = 'ws://127.0.0.1:9998';

/** @type {new (contextOptions?: AudioContextOptions | undefined) => AudioContext} */
let AudioContext,
    /** @type {AudioContext} */
    context,
    /** @type {AudioWorkletNode} */
    processor,
    /** @type {MediaStreamAudioSourceNode} */
    input,
    /** @type {MediaStream} */
    globalStream,
    /** @type {WebSocket} */
    websocket;

const audio = document.getElementById('audio-out');
const mediaSource = new MediaSource();
audio.src = URL.createObjectURL(mediaSource);


//audioStream constraints
const constraints = {
  audio: true,
  video: false,
};

/**
 * Sets up audio context and audio worklet processor
 * @param {Function} onmessage 
 */
async function setUpContext(onmessage) { 
  AudioContext = window.AudioContext || window.webkitAudioContext;
  context = new AudioContext({
    latencyHint: 'interactive'
  });
  await context.audioWorklet.addModule('./recorderWorkletProcessor.js');
  context.resume();
  globalStream = await navigator.mediaDevices.getUserMedia(constraints)
  input = context.createMediaStreamSource(globalStream)
  processor = new window.AudioWorkletNode(
    context,
    'recorder.worklet'
  );
  processor.connect(context.destination);
  context.resume();
  input.connect(processor);
  processor.port.onmessage = onmessage;
}

/**
 * Tears down audio context and audio worklet processor
 */
async function tearDownContext() {
  const track = globalStream.getTracks()[0];
  track.stop();

  input.disconnect(processor);
  processor.disconnect(context.destination);
  await context.close()
  input = null;
  processor = null;
  context = null;
  AudioContext = null;
}

/**
 * Starts recording audio and sends it to the backend
 */
async function startRecordingWS() {
  initWebSocket();
  await setUpContext((e) => {
      const audioData = e.data;
      websocket.send(audioData);
    }
  );
}

/**
 * Stops recording audio and closes the websocket connection
 */
async function stopRecordingWS() {
  tearDownContext();
  websocket.close();
}

/**
 * Prints any (non-empty) messages received from the backend
 * @param {string} line 
 */
function output(line) {
  tx = JSON.parse(line);
  if (tx.tx) {
    currentMessages = document.getElementById("messages").innerHTML;
    document.getElementById("messages").innerHTML = currentMessages + '<br>' + 
    `${tx.start} -> ${tx.end}: ${tx.tx}`;
  }
}

/**
 * Initializes the websocket
 */
function initWebSocket() {
    // Create WebSocket
    websocket = new WebSocket(websocket_uri);     
  
    websocket.onopen = function() {
      console.log("connected to server");
      document.getElementById("webSocketStatus").innerHTML = 'Connected';
    }
    
    websocket.onclose = function(e) {
      console.log("connection closed (" + e.code + ")");
      document.getElementById("webSocketStatus").innerHTML = 'Not Connected';
    }
    
    websocket.onmessage = function(msg) {
      try {
        console.log(msg.data);
        output(msg.data);
      }  catch (e) {
        console.error('Error retrieving data: ' + e);
        console.error(e.stack);
      }
    }
}
