// peer connection
var pc = null;

// text track for subtitles
var textTrack = document.getElementById('video').addTextTrack('captions', 'Smatter');

function createPeerConnection() {
  const config = {
    sdpSemantics: 'unified-plan'
  };
  const pc = new RTCPeerConnection(config);
  const vid = document.getElementById('video');
  vid.addEventListener('loadedmetadata', function(_evt) {
    console.log('Showing text track')
    textTrack.mode = 'showing';
    vid.textTracks[0].mode = 'showing'; // Firefox?
  });
  // connect audio / video
  pc.addEventListener('track', function(evt) {
    console.log(`Got track event ${JSON.stringify(evt)}`)
    if (evt.track.kind == 'video')
      document.getElementById('video').srcObject = evt.streams[0];
    else
      document.getElementById('audio').srcObject = evt.streams[0];
  });
  pc.addEventListener('datachannel', function(evt) {
    console.log(`Connected data channel ${JSON.stringify(evt)}`)
    evt.channel.addEventListener('message', function(evt) {
      console.log(`Got message on data channel`);
      smatterSub = JSON.parse(evt.data);
      console.log(JSON.stringify(smatterSub));
      cue = new VTTCue(smatterSub.start, smatterSub.end, smatterSub.text);
      textTrack.addCue(cue);
    });
  });
  const dc = pc.createDataChannel(
    'smatter_vtt'
  )
  dc.addEventListener('open', function(evt) {
    console.log(`Data channel opened by server ${JSON.stringify(evt)}`)
  });

  return pc;
}

function negotiate() {
  pc.addTransceiver('video', {direction: 'recvonly'});
  pc.addTransceiver('audio', {direction: 'recvonly'});
  return pc.createOffer().then(function(offer) {
    return pc.setLocalDescription(offer);
  }).then(function() {
    var offer = pc.localDescription;

    return fetch('/offer', {
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type
      }),
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST'
    });
  }).then(function(response) {
      return response.json();
  }).then(function(answer) {
      return pc.setRemoteDescription(answer);
  }).catch(function(e) {
      alert(e);
  });
}

function start() {
  pc = createPeerConnection();
  negotiate();
}

start();